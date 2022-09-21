package rddl.competition.generators;

/**
 *  A generator for instances of a fully observable game of life.
 *  
 *  @author Scott Sanner
 *  @version 3/1/11
 * 
 **/

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class GameOfLife2MDPGen {

	protected String output_dir;
	protected String instance_name;
	protected int size_x;
	protected int size_y;
	protected float noise_prob_min;
	protected float noise_prob_max;
	protected float init_state_prob;
	protected int horizon;
	protected float discount;
	protected long seed;
	
	public static void main(String [] args) throws Exception {
		
		if(args.length != 10)
			usage();
		
		GameOfLife2MDPGen gen = new GameOfLife2MDPGen(args);
		String content = gen.generate();
		PrintStream ps = new PrintStream(
				new FileOutputStream(gen.output_dir + File.separator + gen.instance_name + ".rddl"));
		ps.println(content);
		ps.close();
	}
	
	public static void usage() {
		System.err.println("Usage: output-dir instance-name size_x size_y noise-prob-min noise-prob-max init-state-prob horizon discount seed");
		System.err.println("Example: files/testcomp/rddl game_of_life_5_5 5 5 0.1 0.3 0.5 100 0.9 100");
		System.exit(127);
	}
	
	public  GameOfLife2MDPGen(String [] args){
		output_dir = args[0];
		if (output_dir.endsWith("/") || output_dir.endsWith("\\"))
			output_dir = output_dir.substring(0, output_dir.length() - 1);
		
		instance_name = args[1];
		size_x = Integer.parseInt(args[2]);
		size_y = Integer.parseInt(args[3]);
		noise_prob_min = Float.parseFloat(args[4]);
		noise_prob_max = Float.parseFloat(args[5]);
		init_state_prob = Float.parseFloat(args[6]);
		horizon = Integer.parseInt(args[7]);
		discount = Float.parseFloat(args[8]);
		this.seed = Long.parseLong(args[9]);
	}

	public String generate(){

		Random ran = new Random(this.seed);
		StringBuilder sb = new StringBuilder();
		
//		non-fluents game2 {
//
//			domain = game_of_life_mdp;
//					
//			objects { 
//				x_pos : {x1,x2};
//				y_pos : {y1,y2};
//			};
//					  
//			non-fluents { 
//				NEIGHBOR(x1,y1,x1,y2);
//				NEIGHBOR(x1,y1,x2,y1);
//				NEIGHBOR(x1,y1,x2,y2);
//				NOISE-PROB(x1,y1) = 0.5;
//			};
//		}

		sb.append("non-fluents nf_" + instance_name + " {\n");
		sb.append("\tdomain = game_of_life_2_mdp;\n");
		sb.append("\tobjects {\n");

		sb.append("\t\tlocation : {");
		for (int i = 1; i <= size_x; i++) {
			
			for (int j = 1; j <= size_y; j++) {
				
				if (i > 1 || j > 1) {
					
					sb.append(",");
				}
				
				sb.append("l" + i + j);
			}
		}
		sb.append("};\n");
		
		
		sb.append("\t};\n");
		
		sb.append("\tnon-fluents {\n");
		
		for (int i = 1; i <= size_x; i++)
			for (int j = 1; j <= size_y; j++) {
				sb.append("\t\tNOISE-PROB(l" + i + j + ") = " + 
						(noise_prob_min + (noise_prob_max - noise_prob_min)*ran.nextFloat())+ ";\n");
				for (int io = -1; io <= 1; io++)
					for (int jo = -1; jo <= 1; jo++) {
						if (io == 0 && jo == 0)
							continue;
						int xn = i + io;
						int yn = j + jo;
						if (xn < 1 || xn > size_x || yn < 1 || yn > size_y)
							continue;
						sb.append("\t\tNEIGHBOR(l" + i + j + ",l" + xn + yn + ");\n");
					}
			}
		
		sb.append("\t};\n");
		sb.append("}\n\n");
		
//		instance glm1 {
//			domain = game_of_life_mdp;	
//			non-fluents = game2;
//			init-state { 
//				alive(x1,y1); 
//				alive(x1,y2); 
//			};
//		  
//			max-nondef-actions = 1;
//			horizon  = 20;
//			discount = 1.0;
//		}

		boolean first = true;
		sb.append("instance " + instance_name + " {\n");
		sb.append("\tdomain = game_of_life_2_mdp;\n");
		sb.append("\tnon-fluents = nf_" + instance_name + ";\n");
		sb.append("\tinit-state {\n");
		for (int i = 1; i <= size_x; i++)
			for (int j = 1; j <= size_y; j++)
				if (first || ran.nextFloat() < init_state_prob) {
					
					first = false;
					sb.append("\t\talive(l" + i + j + ");\n");
				}
					
		sb.append("\t};\n\n");
		sb.append("\tmax-nondef-actions = 1;\n");
		sb.append("\thorizon  = " + horizon + ";\n");
		sb.append("\tdiscount = " + discount + ";\n");
		
		sb.append("}");
		
		return sb.toString();
	}
	
}
