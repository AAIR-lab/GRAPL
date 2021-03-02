import numpy as np
import copy
import torch
import time
from tqdm import tqdm
from torchviz import make_dot
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import remove_self_loops, add_self_loops
from sklearn.metrics import accuracy_score

tqdm().pandas()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def categorical_accuracy(y_pred, y_true):
    _, pred = torch.max(torch.nn.functional.softmax(y_pred,1), dim=1)
    y_true = y_true.cpu().numpy()
    y_pred = pred.detach().cpu().numpy()
    return torch.tensor(accuracy_score(y_true, y_pred), dtype=torch.float)

def binary_accuracy_multi(y_pred, y_true):
    y_pred = torch.sigmoid(y_pred)
    y_pred = y_pred.detach().cpu().numpy()
    y_pred = np.where(y_pred>=0.5,y_pred,0)
    y_pred = np.where(y_pred<=0.5,y_pred,1)
    y_true = y_true.cpu().numpy()
    return torch.tensor(accuracy_score(y_true, y_pred), dtype=torch.float)


def get_trainer(model, phase, dataloader, output_dict, print_params=False, report_freq=10, writer_dir="/tmp/runs/m1"):
    dataloaders = {phase: dataloader}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.007)
    crit1 = torch.nn.CrossEntropyLoss()
    crit2 = torch.nn.BCEWithLogitsLoss()
    names = list(output_dict.keys())
    criteria = {names[0]: crit1, names[1]: crit2, names[2]: crit2}
    metrics = {names[0]: categorical_accuracy, names[1]: binary_accuracy_multi, names[2]: binary_accuracy_multi}
    scheduler = None #torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.001)
    writer = SummaryWriter(writer_dir)

    return PyTorch_Backend(dataloaders=dataloaders, optimizer=optimizer, criteria=criteria, metrics=metrics,
                           scheduler=scheduler, report_freq=report_freq, print_params=print_params, writer=writer)


def model_summary(model):
    s = 0
    print("=" * 100)
    print("|" + str.center('Layer Name', 53) + "|" + str.center('Number of Parameters', 44) + "|")
    print("=" * 100)
    for name, params in model.state_dict().items():
        print("|" + str.ljust(name, 53) + "|" + str.rjust(f'{torch.numel(params):,}', 44) + "|")
        s += torch.numel(params)
    print("=" * 100)
    print("|" + str.center("Total Parameters", 53) + "|" + str.rjust(f'{s:,}', 44) + "|")
    print("=" * 100)


class PyTorch_Backend:

    def __init__(self, dataloaders, optimizer, criteria, metrics, scheduler, report_freq, print_params, writer):
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.criteria = criteria
        self.metrics = metrics
        self.scheduler = scheduler
        self.report_freq = report_freq
        self.print_params = print_params
        self.writer = writer

    def train(self, model, epochs=30):
        start = time.time()
        # Load model on GPU
        if torch.cuda.is_available():
            model = model.cuda()

        # Print the trainable parameters of the network
        if self.print_params:
            for name, param in model.state_dict().items():
                print(name, param.size())

            # Getting a backward plot of the model
            for k in self.dataloaders['train']:
                o = model(k.to(device))
                break
            print("Params")
            for key in o:
                g = make_dot(o[key])
            g.format = 'png'
            g.view('model', '/tmp')

        # Training epochs
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)

            # For each phase we, get the model output for calculating loss 
            for phase in self.dataloaders.keys():

                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                epoch_loss, epoch_metrics = {k: [] for k in self.criteria}, {k: [] for k in self.metrics}
                report_loss, report_metrics = {k: [] for k in self.criteria}, {k: [] for k in self.metrics}

                for i, item in enumerate(self.dataloaders[phase]):
                    item = item.to(device)
                    out = item.y
                    from itertools import starmap

                    # Convert list of outputs to output of lists
                    new_out = {}
                    for k in self.criteria:
                        if k == "action":
                            new_out[k] = torch.tensor(np.array(list(starmap(lambda x: x[k], out)), dtype=np.long), dtype=torch.long).to(device)
                        else:
                            new_out[k] = torch.tensor(np.array(list(starmap(lambda x: x[k], out)), dtype=np.float32),
                                                      dtype=torch.float32).to(device)
                    out = copy.deepcopy(new_out)
                    del new_out
                    # Convert list of outputs to dictionary of lists

                    self.optimizer.zero_grad()

                    loss = 0
                    with torch.set_grad_enabled(phase == 'train'):
                        pred = model(item)
                        losses, metric = {}, {}
                        for k in self.criteria:
                            losses[k] = self.criteria[k](pred[k], out[k])
                            report_loss[k].append(losses[k].item())
                            epoch_loss[k].append(losses[k].item())
                            if self.writer is not None:
                                self.writer.add_scalar('Batch-wise ' + phase + ' Loss ' + k, losses[k],
                                                       epoch * len(self.dataloaders[phase]) + i)
                            loss += losses[k] / len(self.criteria)
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                            if self.scheduler:
                                self.scheduler.step(loss)

                    # Get metric scores
                    for k in self.metrics:
                        metric[k] = self.metrics[k](pred[k], out[k])
                        report_metrics[k].append(metric[k])
                        epoch_metrics[k].append(metric[k])
                        if self.writer is not None:
                            self.writer.add_scalar('Batch-wise ' + phase + ' Metric ' + k, metric[k],
                                                   epoch * len(self.dataloaders[phase]) + i)

                    if self.writer is not None:
                        for name, param in model.state_dict().items():
                            self.writer.add_histogram(name, param, epoch * len(self.dataloaders[phase]) + i)
                        for k in self.metrics:
                            self.writer.add_histogram(k, F.softmax(pred[k],1), epoch * len(self.dataloaders[phase]) + i)

                    if i % self.report_freq == 0:
                        # Prepare statement to print
                        cnt = 2
                        statement = "\033[1m Phase \033[0m {0:5s} \033[1m Epoch \033[0m {1:3d} \033[1m Batch \033[0m{" \
                                    "2:3d} \033[1m Losses \033[0m "
                        content = (phase, epoch, i)
                        for k in self.criteria:
                            cnt += 1
                            statement += k + " {" + str(cnt) + ":3.3f} "
                            content += (torch.mean(torch.tensor(report_loss[k])).item(),)
                            report_loss[k] = []
                        statement += " \033[1m Metrics \033[0m"
                        for k in self.metrics:
                            cnt += 1
                            statement += k + " {" + str(cnt) + ":3.3f} "
                            content += (torch.mean(torch.tensor(report_metrics[k])).item(),)
                            report_metrics[k] = []
                        print(statement.format(*content))

                # Prepare statement to print
                print('-' * 70)
                cnt = 1
                statement = "\033[1m Phase \033[0m {0:5s} \033[1m Epoch \033[0m {1:3d} \033[1m Losses \033[0m"
                content = (phase, epoch,)
                for k in self.criteria:
                    cnt += 1
                    statement += k + " {" + str(cnt) + ":3.3f} "
                    content += (torch.mean(torch.tensor(epoch_loss[k])).item(),)
                    if self.writer is not None:
                        self.writer.add_scalar('Epoch-end ' + phase + ' Loss ' + k,
                                               torch.mean(torch.tensor(epoch_loss[k])).item(), epoch)
                    epoch_loss[k] = []
                statement += " \033[1m Metrics \033[0m"
                for k in self.metrics:
                    cnt += 1
                    statement += k + " {" + str(cnt) + ":3.3f} "
                    content += (torch.mean(torch.tensor(epoch_metrics[k])).item(),)
                    if self.writer is not None:
                        self.writer.add_scalar('Epoch-end ' + phase + ' Metric ' + k,
                                               torch.mean(torch.tensor(epoch_metrics[k])).item(), epoch)
                    epoch_metrics[k] = []
                print(statement.format(*content))
                print('-' * 70)
        print(time.time()-start)
        return model

    def eval(self, model, phase='eval'):
        # Load model on GPU
        if torch.cuda.is_available():
            model = model.cuda()

        model.eval()

        epoch_loss, epoch_metrics = {k: [] for k in self.criteria}, {k: [] for k in self.metrics}
        report_loss, report_metrics = {k: [] for k in self.criteria}, {k: [] for k in self.metrics}

        for i, item in enumerate(self.dataloaders[phase]):
            item = item.to(device)
            out = item.y
            from itertools import starmap

            # Convert list of outputs to output of lists
            new_out = {}
            for k in self.criteria:
                if k == "action":
                    new_out[k] = torch.tensor(np.array(list(starmap(lambda x: x[k], out)), dtype=np.long), dtype=torch.long).to(device)
                else:
                    new_out[k] = torch.tensor(np.array(list(starmap(lambda x: x[k], out)), dtype=np.float32),
                                                dtype=torch.float32).to(device)
            out = copy.deepcopy(new_out)
            del new_out
            # Convert list of outputs to dictionary of lists

            self.optimizer.zero_grad()

            loss = 0
            with torch.set_grad_enabled(phase == 'train'):
                pred = model(item)
                losses, metric = {}, {}
                for k in self.criteria:
                    losses[k] = self.criteria[k](pred[k], out[k])
                    report_loss[k].append(losses[k].item())
                    epoch_loss[k].append(losses[k].item())
                    if self.writer is not None:
                        self.writer.add_scalar('Batch-wise ' + phase + ' Loss ' + k, losses[k],
                                                0 * len(self.dataloaders[phase]) + i)
                    loss += losses[k] / len(self.criteria)
                    
            # Get metric scores
            for k in self.metrics:
                metric[k] = self.metrics[k](pred[k], out[k])
                report_metrics[k].append(metric[k])
                epoch_metrics[k].append(metric[k])
                if self.writer is not None:
                    self.writer.add_scalar('Batch-wise ' + phase + ' Metric ' + k, metric[k],
                                            0 * len(self.dataloaders[phase]) + i)

            if self.writer is not None:
                for name, param in model.state_dict().items():
                    self.writer.add_histogram(name, param, 0 * len(self.dataloaders[phase]) + i)
                for k in self.metrics:
                    self.writer.add_histogram(k, F.softmax(pred[k],1), 0 * len(self.dataloaders[phase]) + i)

            if i % self.report_freq == 0:
                # Prepare statement to print
                cnt = 2
                statement = "\033[1m Phase \033[0m {0:5s} \033[1m Epoch \033[0m {1:3d} \033[1m Batch \033[0m{" \
                            "2:3d} \033[1m Losses \033[0m "
                content = (phase, epoch, i)
                for k in self.criteria:
                    cnt += 1
                    statement += k + " {" + str(cnt) + ":3.3f} "
                    content += (torch.mean(torch.tensor(report_loss[k])).item(),)
                    report_loss[k] = []
                statement += " \033[1m Metrics \033[0m"
                for k in self.metrics:
                    cnt += 1
                    statement += k + " {" + str(cnt) + ":3.3f} "
                    content += (torch.mean(torch.tensor(report_metrics[k])).item(),)
                    report_metrics[k] = []
                print(statement.format(*content))

        # Prepare statement to print
        print('-' * 70)
        cnt = 1
        statement = "\033[1m Phase \033[0m {0:5s} \033[1m Epoch \033[0m {1:3d} \033[1m Losses \033[0m"
        content = (phase, 0,)
        for k in self.criteria:
            cnt += 1
            statement += k + " {" + str(cnt) + ":3.3f} "
            content += (torch.mean(torch.tensor(epoch_loss[k])).item(),)
            if self.writer is not None:
                self.writer.add_scalar('Epoch-end ' + phase + ' Loss ' + k,
                                        torch.mean(torch.tensor(epoch_loss[k])).item(), 0)
            epoch_loss[k] = []
        statement += " \033[1m Metrics \033[0m"
        for k in self.metrics:
            cnt += 1
            statement += k + " {" + str(cnt) + ":3.3f} "
            content += (torch.mean(torch.tensor(epoch_metrics[k])).item(),)
            if self.writer is not None:
                self.writer.add_scalar('Epoch-end ' + phase + ' Metric ' + k,
                                        torch.mean(torch.tensor(epoch_metrics[k])).item(), 0)
            epoch_metrics[k] = []
        print(statement.format(*content))
        print('-' * 70)

    def predict(self, model, phase='test'):
        device = 'cpu'
        model.eval()
        predictions = {k: [] for k in self.criteria}
        for i, item in enumerate(self.dataloaders[phase]):
            item = item.to(device)
            pred = model(item)
            for k in self.criteria:
                predictions[k].extend(pred[k].detach().cpu().numpy().tolist())
        for k in self.criteria:
            predictions[k] = np.array(predictions[k])
        return predictions


class DAGConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(DAGConv, self).__init__(aggr='add', flow='source_to_target')
        self.lin1 = torch.nn.Linear(in_channels, out_channels)
        self.lin2 = torch.nn.Linear(in_channels, out_channels)
        self.act = torch.nn.ReLU()

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j):
        x_i = self.act(self.lin1(x_i))
        x_j = self.act(self.lin2(x_j))
        return x_i + x_j

    def update(self, aggr_out):
        return aggr_out
