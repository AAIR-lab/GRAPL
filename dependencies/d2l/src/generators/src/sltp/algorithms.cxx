
#include <limits>
#include <deque>
#include <vector>
#include <cassert>

#include <sltp/dl/types.hxx>
#include <sltp/algorithms.hxx>

namespace sltp {

int compute_distance(unsigned num_objects,
                     const state_denotation_t &start_sd,
                     const state_denotation_t &end_sd,
                     const state_denotation_t &role_sd) {
    // create adjacency lists
    std::vector<std::vector<int> > adj(num_objects);
    for( int i = 0; i < num_objects; ++i ) {
        for( int j = 0; j < num_objects; ++j ) {
            if( role_sd[i * num_objects + j] )
                adj[i].emplace_back(j);
        }
    }

    // locate start vertex
    int start = -1;
    for( int i = 0; i < num_objects; ++i ) {
        if( start_sd[i] ) {
            start = i;
            break;
        }
    }
    assert(start != -1);

    // check whether distance is 0
    if( end_sd[start] ) return 0;

    // apply breadth-first search from start vertex
    std::vector<int> distances(num_objects, -1);
    distances[start] = 0;

    std::deque<std::pair<int, int> > q;
    for (const auto& obj: adj[start]) {
        if( end_sd[obj] ) return 1;
        else q.emplace_back(obj, 1);
    }

    while( !q.empty() ) {
        std::pair<int, int> p = q.front();
        q.pop_front();
        if( distances[p.first] == -1 ) {
            distances[p.first] = p.second;
            for( int i = 0; i < adj[p.first].size(); ++i ) {
                if( end_sd[adj[p.first][i]] )
                    return 1 + p.second;
                else
                    q.emplace_back(adj[p.first][i], 1 + p.second);
            }
        }
    }
    return std::numeric_limits<int>::max();
}

} // namespaces