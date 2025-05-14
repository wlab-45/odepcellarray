from math import fabs
from itertools import combinations
from copy import deepcopy
import math
import heapq # 確保導入 heapq
import multiprocessing # Add this import

# At the top level of your cbs.py, or in a helper module
def run_astar_for_agent_process(args):
    """
    Worker function for multiprocessing. Executes A* for a single agent.
    Args:
        args (tuple): Contains all necessary data for A* search:
            agent_name,
            start_state_tuple (time, x, y),
            goal_location_tuple (x, y),
            dimension_tuple (cols, rows),
            obstacles_set_tuples (set of (x,y)),
            agent_specific_constraints_tuple ( (vc_tuples), (ec_tuples) )
    Returns:
        (agent_name, path_list_of_state_tuples or None)
    """
    agent_name, start_state_tuple, goal_location_tuple, \
    dimension_tuple, obstacles_set_tuples, agent_specific_constraints_tuple = args

    # Reconstruct objects needed for A* inside the new process
    # This is crucial because objects from the parent process might not be directly usable
    # or could lead to issues if they contain non-picklable state (like open file handles, locks etc.)

    # Create a minimal 'mock' or 'partial' Environment or pass data directly to AStar
    # For simplicity here, let's assume AStar can be instantiated with direct data or a simplified env.
    
    start_loc = Location(start_state_tuple[1], start_state_tuple[2])
    start_state = State(start_state_tuple[0], start_loc)
    goal_loc = Location(goal_location_tuple[0], goal_location_tuple[1])

    # Reconstruct constraints
    agent_constraints = Constraints()
    for vc_data in agent_specific_constraints_tuple[0]: # Vertex Constraints
        # vc_data should be (time, loc_x, loc_y)
        agent_constraints.add_constraint(VertexConstraint(vc_data[0], Location(vc_data[1], vc_data[2])))
    for ec_data in agent_specific_constraints_tuple[1]: # Edge Constraints
        # ec_data should be (time, loc1_x, loc1_y, loc2_x, loc2_y)
        agent_constraints.add_constraint(EdgeConstraint(ec_data[0],
                                                       Location(ec_data[1], ec_data[2]),
                                                       Location(ec_data[3], ec_data[4])))

    # Simplified environment data for this A* instance
    # This avoids pickling the whole Environment object
    class MiniEnvForAStar:
        def __init__(self, dim, obs, ag_dict, ag_constr, current_ag_name):
            self.dimension = dim
            self.obstacles = obs
            self.agent_dict = {current_ag_name: ag_dict} # Only info for the current agent
            self.current_hl_node_constraints = {current_ag_name: ag_constr} # Agent's specific constraints
            self.visiting_agent_name = current_ag_name # For state_valid context

        def get_agent_constraints(self, agent_name):
            return self.current_hl_node_constraints.get(agent_name, Constraints())

        def heuristic(self, current_state, agent_name_param):
            goal_loc_for_heuristic = self.agent_dict[agent_name_param]["goal_location"]
            return abs(current_state.location.x - goal_loc_for_heuristic.x) + \
                   abs(current_state.location.y - goal_loc_for_heuristic.y)

        def is_at_goal(self, current_state, agent_name_param):
            return current_state.location == self.agent_dict[agent_name_param]["goal_location"]

        def state_valid(self, state, visiting_agent_name_param, agent_specific_constraints_param):
            if not (0 <= state.location.x < self.dimension[0] and \
                    0 <= state.location.y < self.dimension[1]):
                return False
            if VertexConstraint(state.time, state.location) in agent_specific_constraints_param.vertex_constraints:
                return False
            
            is_on_general_obstacle = (state.location.x, state.location.y) in self.obstacles
            if is_on_general_obstacle:
                agent_start_loc = self.agent_dict[visiting_agent_name_param]['start'].location
                if state.location == agent_start_loc and state.time == 0:
                    return True
                return False
            return True

        def transition_valid(self, state_1, state_2, agent_specific_constraints_param):
            if EdgeConstraint(state_1.time, state_1.location, state_2.location) in agent_specific_constraints_param.edge_constraints:
                return False
            return True
        
        def get_neighbors(self, current_state, visiting_agent_name_param):
            # This is a copy of the original Environment.get_neighbors, but using self.
            neighbors = []
            possible_moves = [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]
            agent_constraints_for_neigh = self.get_agent_constraints(visiting_agent_name_param)

            for dx, dy in possible_moves:
                new_loc = Location(current_state.location.x + dx, current_state.location.y + dy)
                next_state = State(current_state.time + 1, new_loc)
                if self.state_valid(next_state, visiting_agent_name_param, agent_constraints_for_neigh) and \
                   self.transition_valid(current_state, next_state, agent_constraints_for_neigh):
                    neighbors.append(next_state)
            return neighbors


    # Prepare agent_dict for MiniEnv
    agent_dict_for_astar = {
        'start': start_state,
        'goal_location': goal_loc
    }
    
    mini_env = MiniEnvForAStar(dimension_tuple, obstacles_set_tuples, agent_dict_for_astar, agent_constraints, agent_name)
    astar_solver = AStar(mini_env) # AStar now takes this mini_env
    
    path_states = astar_solver.search(agent_name)

    if path_states:
        # Convert path_states (list of State objects) to list of serializable tuples for return
        # e.g., [(time, x, y), ...]
        serializable_path = []
        for s_obj in path_states:
            serializable_path.append((s_obj.time, s_obj.location.x, s_obj.location.y))
        return agent_name, serializable_path
    else:
        return agent_name, None
    
class Location(object):
    def __init__(self, x=-1, y=-1):
        self.x = x
        self.y = y

    def __eq__(self, other):
        if not isinstance(other, Location):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __str__(self):
        return str((self.x, self.y))

class State(object):
    def __init__(self, time, location):
        self.time = time
        self.location = location # Location object

    def __eq__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        return self.time == other.time and self.location == other.location

    def __hash__(self):
        # Ensure location is properly hashed if it's a custom object
        return hash((self.time, self.location.x, self.location.y))

    def is_equal_except_time(self, other_state):
        if not isinstance(other_state, State):
            return False
        return self.location == other_state.location

    def __str__(self):
        return str((self.time, self.location.x, self.location.y))

    def __lt__(self, other): # For heapq when f_scores are equal
        if self.time != other.time:
            return self.time < other.time
        if self.location.x != other.location.x:
            return self.location.x < other.location.x
        return self.location.y < other.location.y


class Conflict(object):
    VERTEX = 1
    EDGE = 2
    def __init__(self):
        self.time = -1
        self.type = -1
        self.agent_1 = '' # name
        self.agent_2 = '' # name
        self.location_1 = Location() # For VERTEX, location_1 is the conflict spot. For EDGE, agent_1 moves from loc_1 to loc_2
        self.location_2 = Location() # For EDGE, agent_1 moves from loc_1 to loc_2 (loc_2 is where agent_1 is at t+1)

    def __str__(self):
        if self.type == Conflict.VERTEX:
            return f"(T={self.time}, VertexConflict, Agents:({self.agent_1}, {self.agent_2}), Loc:{self.location_1})"
        elif self.type == Conflict.EDGE:
            return f"(T={self.time}, EdgeConflict, Agents:({self.agent_1}, {self.agent_2}), Path1:({self.location_1}->{self.location_2}), Path2:({self.location_2}->{self.location_1}))"
        return "Invalid Conflict"


class VertexConstraint(object):
    def __init__(self, time, location): # location is a Location object
        self.time = time
        self.location = location

    def __eq__(self, other):
        if not isinstance(other, VertexConstraint):
            return NotImplemented
        return self.time == other.time and self.location == other.location

    def __hash__(self):
        return hash((self.time, self.location)) # Location object should be hashable

    def __str__(self):
        return f'(VC T:{self.time}, Loc:{self.location})'

class EdgeConstraint(object):
    def __init__(self, time, location_1, location_2): # location_1, location_2 are Location objects
        self.time = time # time of arriving at location_1 (start of edge)
        self.location_1 = location_1 # from
        self.location_2 = location_2 # to

    def __eq__(self, other):
        if not isinstance(other, EdgeConstraint):
            return NotImplemented
        return self.time == other.time and \
               self.location_1 == other.location_1 and \
               self.location_2 == other.location_2

    def __hash__(self):
        return hash((self.time, self.location_1, self.location_2))

    def __str__(self):
        return f'(EC T:{self.time}, Edge:{self.location_1}->{self.location_2})'

class Constraints(object):
    def __init__(self):
        self.vertex_constraints = set() # Set of VertexConstraint objects
        self.edge_constraints = set()   # Set of EdgeConstraint objects

    def add_constraint(self, other_constraints):
        if isinstance(other_constraints, Constraints): # If merging another Constraints object
            self.vertex_constraints.update(other_constraints.vertex_constraints)
            self.edge_constraints.update(other_constraints.edge_constraints)
        # Allow adding single constraints too (optional)
        elif isinstance(other_constraints, VertexConstraint):
            self.vertex_constraints.add(other_constraints)
        elif isinstance(other_constraints, EdgeConstraint):
            self.edge_constraints.add(other_constraints)


    def __str__(self):
        vc_str = ", ".join(map(str, sorted(list(self.vertex_constraints), key=lambda x: (x.time, x.location.x, x.location.y))))
        ec_str = ", ".join(map(str, sorted(list(self.edge_constraints), key=lambda x: (x.time, x.location_1.x, x.location_1.y, x.location_2.x, x.location_2.y))))
        return f"VCs: {{{vc_str}}} ECs: {{{ec_str}}}"


class Environment(object):
    def __init__(self, dimension, agents_data, obstacles_grid):
        self.dimension = dimension  # (num_cols, num_rows)
        self.obstacles = obstacles_grid  # set of (x,y) grid tuples for general obstacles
        self.agents_data_input = agents_data # Initial agent data list

        self.agent_dict = {}  # agent_name -> {'start': State, 'goal': State}
        self.make_agent_dict()

        # This will be set by the HighLevelNode before its solution is computed
        self.current_hl_node_constraints = {} # agent_name -> Constraints object
        
        self.a_star = AStar(self)

    def make_agent_dict(self):
        for agent_info in self.agents_data_input:
            start_loc = Location(agent_info['start'][0], agent_info['start'][1])
            goal_loc = Location(agent_info['goal'][0], agent_info['goal'][1])
            # Goal time is not fixed, agent reaches goal at any time t >= 0
            # Store goal location, is_at_goal will check location equality.
            self.agent_dict[agent_info['name']] = {
                'start': State(0, start_loc),
                'goal_location': goal_loc # Store goal location, not a full State
            }
    
    def get_agent_constraints(self, agent_name):
        """Helper to get constraints for a specific agent for the current HL node."""
        return self.current_hl_node_constraints.get(agent_name, Constraints())

    def get_neighbors(self, current_state, visiting_agent_name):
        neighbors = []
        # Order: Wait, N, E, S, W (or any consistent order)
        possible_moves = [
            (0, 0),  # Wait
            (0, 1),  # Up (y increases)
            (1, 0),  # Right (x increases)
            (0, -1), # Down (y decreases)
            (-1, 0)  # Left (x decreases)
        ]

        for dx, dy in possible_moves:
            new_loc = Location(current_state.location.x + dx, current_state.location.y + dy)
            next_state = State(current_state.time + 1, new_loc)
            
            agent_specific_constraints = self.get_agent_constraints(visiting_agent_name)

            if self.state_valid(next_state, visiting_agent_name, agent_specific_constraints) and \
               self.transition_valid(current_state, next_state, agent_specific_constraints):
                neighbors.append(next_state)
        return neighbors

    def get_state(self, agent_name, solution_dict, time_step):
        agent_path = solution_dict.get(agent_name)
        if not agent_path: # Should not happen if solution is valid for this agent
            # This case needs careful handling. Returning a dummy far-off state or raising error.
            # For now, assume agent_path exists if agent_name is in solution_dict
            return State(time_step, Location(-1,-1)) # Dummy, indicates error or missing path

        if time_step < len(agent_path):
            return agent_path[time_step]
        else:
            # Agent stays at its goal location indefinitely after reaching it
            return State(time_step, agent_path[-1].location)


    def get_first_conflict(self, current_solution_dict):
        if not current_solution_dict:
            return None

        max_time = 0
        for agent_name in current_solution_dict:
            path = current_solution_dict[agent_name]
            if path: # Ensure path is not None or empty
                 max_time = max(max_time, len(path) -1) # Max index is len-1
            else: # If an agent has no path, solution_dict is malformed for conflict checking
                return None 

        # Check for Vertex Conflicts (agents occupying the same cell at the same time)
        for t in range(max_time + 1): # Iterate through each time step up to max_time
            occupied_locations_at_t = {} # loc_hash -> list_of_agent_names
            for agent_name in current_solution_dict:
                agent_state_at_t = self.get_state(agent_name, current_solution_dict, t)
                
                # Skip t=0 vertex conflicts if agents start at the same location (as per problem specific rules)
                # However, standard CBS usually aims to resolve all spatio-temporal conflicts.
                # If the rule is "agents can start at same cell, but must move apart",
                # then a t=0 conflict is not a "CBS conflict" to branch on.
                # But if they must occupy distinct cells from t=0, then it is.
                # Let's assume for now t=0 vertex conflicts ARE NOT branched on,
                # if multiple agents start at the same cell.
                # This should be handled by your problem setup or a pre-processing step
                # if starting cells must be unique.
                # For now, strict CBS: if two agents are at same loc at same time (t>0), it's a conflict.
                
                # For t=0, if multiple agents are at the same start cell, this is allowed by problem rules,
                # so we don't treat it as a conflict for CBS branching.
                if t == 0 and len([name for name in current_solution_dict if self.get_state(name, current_solution_dict, 0).location == agent_state_at_t.location]) > 1:
                     pass # Multiple agents at same start spot at t=0, allowed.

                loc_tuple = (agent_state_at_t.location.x, agent_state_at_t.location.y)
                if loc_tuple not in occupied_locations_at_t:
                    occupied_locations_at_t[loc_tuple] = []
                occupied_locations_at_t[loc_tuple].append(agent_name)

            for loc_tuple, agents_in_loc in occupied_locations_at_t.items():
                if len(agents_in_loc) > 1:
                    # Found a vertex conflict
                    # Skip if it's a t=0 conflict at a shared start location (if this is an allowed scenario)
                    is_shared_start = False
                    if t == 0:
                        start_loc_obj = Location(loc_tuple[0], loc_tuple[1])
                        num_starting_here = 0
                        for ag_name in agents_in_loc:
                            if self.agent_dict[ag_name]['start'].location == start_loc_obj:
                                num_starting_here +=1
                        if num_starting_here == len(agents_in_loc) and len(agents_in_loc) > 1 : # All conflicting agents started here
                            is_shared_start = True
                    
                    if not is_shared_start: # Only consider it a conflict if not a shared start at t=0
                        conflict = Conflict()
                        conflict.time = t
                        conflict.type = Conflict.VERTEX
                        conflict.agent_1 = agents_in_loc[0]
                        conflict.agent_2 = agents_in_loc[1]
                        conflict.location_1 = Location(loc_tuple[0], loc_tuple[1])
                        return conflict

        # Check for Edge Conflicts (agents swapping cells)
        for t in range(max_time): # Transitions from t to t+1
            for agent1_name, agent2_name in combinations(current_solution_dict.keys(), 2):
                path1 = current_solution_dict[agent1_name]
                path2 = current_solution_dict[agent2_name]
                if not path1 or not path2: continue


                pos1_t0 = self.get_state(agent1_name, current_solution_dict, t).location
                pos1_t1 = self.get_state(agent1_name, current_solution_dict, t + 1).location
                
                pos2_t0 = self.get_state(agent2_name, current_solution_dict, t).location
                pos2_t1 = self.get_state(agent2_name, current_solution_dict, t + 1).location

                if pos1_t0 == pos2_t1 and pos1_t1 == pos2_t0:
                    conflict = Conflict()
                    conflict.time = t # Conflict occurs over interval t to t+1
                    conflict.type = Conflict.EDGE
                    conflict.agent_1 = agent1_name
                    conflict.agent_2 = agent2_name
                    conflict.location_1 = pos1_t0 # Agent1's location at time t
                    conflict.location_2 = pos1_t1 # Agent1's location at time t+1
                    return conflict
        return None # No conflicts found

    def create_constraints_from_conflict(self, conflict):
        constraints_to_add = {} # agent_name -> Constraints object with the new constraint
        
        agent1_constraints = Constraints()
        agent2_constraints = Constraints()

        if conflict.type == Conflict.VERTEX:
            # Add a vertex constraint for both agents at the conflict location and time
            vc = VertexConstraint(conflict.time, conflict.location_1)
            agent1_constraints.add_constraint(vc)
            agent2_constraints.add_constraint(vc)
            constraints_to_add[conflict.agent_1] = agent1_constraints
            constraints_to_add[conflict.agent_2] = agent2_constraints

        elif conflict.type == Conflict.EDGE:
            # Agent 1 is constrained from moving loc1 -> loc2 at time t
            # Agent 2 is constrained from moving loc2 -> loc1 at time t
            # Conflict.location_1 was agent1's loc at t. Conflict.location_2 was agent1's loc at t+1.
            ec1 = EdgeConstraint(conflict.time, conflict.location_1, conflict.location_2) # For agent1
            agent1_constraints.add_constraint(ec1)
            constraints_to_add[conflict.agent_1] = agent1_constraints

            # Agent2 was at conflict.location_2 (agent1's t+1 pos) at time t,
            # and moved to conflict.location_1 (agent1's t pos) at time t+1.
            ec2 = EdgeConstraint(conflict.time, conflict.location_2, conflict.location_1) # For agent2
            agent2_constraints.add_constraint(ec2)
            constraints_to_add[conflict.agent_2] = agent2_constraints
            
        return constraints_to_add

    def state_valid(self, state, visiting_agent_name, agent_specific_constraints):
        # 1. Check bounds
        if not (0 <= state.location.x < self.dimension[0] and
                0 <= state.location.y < self.dimension[1]):
            return False

        # 2. Check agent-specific vertex constraints
        if VertexConstraint(state.time, state.location) in agent_specific_constraints.vertex_constraints:
            return False

        # 3. Check general obstacles, with special rule for visiting_agent_name's start
        is_on_general_obstacle = (state.location.x, state.location.y) in self.obstacles
        
        if is_on_general_obstacle:
            agent_start_loc = self.agent_dict[visiting_agent_name]['start'].location
            if state.location == agent_start_loc and state.time == 0:
                return True  # Valid for this agent at t=0 at its own start obstacle cell
            return False # Invalid if on an obstacle otherwise
            
        return True # Valid if not out of bounds, not constrained, and not on an (un-exempted) obstacle

    def transition_valid(self, state_1, state_2, agent_specific_constraints):
        # Check agent-specific edge constraints
        if EdgeConstraint(state_1.time, state_1.location, state_2.location) in agent_specific_constraints.edge_constraints:
            return False
        return True

    def heuristic(self, current_state, agent_name): # Removed unused parameters
        goal_loc = self.agent_dict[agent_name]["goal_location"]
        # Manhattan distance
        return abs(current_state.location.x - goal_loc.x) + \
               abs(current_state.location.y - goal_loc.y)

    def is_at_goal(self, current_state, agent_name):
        # Agent is at goal if its current location matches the goal location.
        # Time doesn't matter for being "at goal", agent can wait at goal.
        return current_state.location == self.agent_dict[agent_name]["goal_location"]


    def compute_solution_for_node(self, constraint_dict_for_node):
        self.current_hl_node_constraints = constraint_dict_for_node
        
        tasks_args = []
        for agent_name, agent_info in self.agent_dict.items():
            start_st = agent_info['start']
            goal_loc = agent_info['goal_location']
            agent_constraints = self.current_hl_node_constraints.get(agent_name, Constraints())

            # Prepare serializable constraint data
            vc_tuples_serializable = tuple([(vc.time, vc.location.x, vc.location.y) for vc in agent_constraints.vertex_constraints])
            ec_tuples_serializable = tuple([(ec.time, ec.location_1.x, ec.location_1.y, ec.location_2.x, ec.location_2.y) for ec in agent_constraints.edge_constraints])
            
            serializable_constraints = (vc_tuples_serializable, ec_tuples_serializable)

            tasks_args.append((
                agent_name,
                (start_st.time, start_st.location.x, start_st.location.y), # Serializable start
                (goal_loc.x, goal_loc.y), # Serializable goal
                self.dimension, # tuple (cols, rows)
                self.obstacles, # set of (x,y) tuples
                serializable_constraints
            ))

        solution_dict = {}
        # Determine number of processes (e.g., number of CPU cores)
        # Be careful not to create too many processes if num_agents is very large.
        # num_processes = min(multiprocessing.cpu_count(), len(self.agent_dict))
        num_processes = multiprocessing.cpu_count() # Or a fixed number like 4 or 8

        try:
            # Using a context manager for the pool is good practice
            with multiprocessing.Pool(processes=num_processes) as pool:
                # starmap is good if args for each task are tuples of arguments
                # results is a list of (agent_name, serializable_path or None)
                results = pool.map(run_astar_for_agent_process, tasks_args)
            
            for agent_name_res, serializable_path_res in results:
                if serializable_path_res is None:
                    return None # One agent failed, so this HL node has no solution
                
                # Deserialize path back to State objects
                path_of_states = []
                for t, x, y in serializable_path_res:
                    path_of_states.append(State(t, Location(x,y)))
                solution_dict[agent_name_res] = path_of_states
            
            return solution_dict

        except Exception as e:
            print(f"Multiprocessing A* error: {e}")
            # Fallback or error handling
            # For simplicity, let's try serial execution if multiprocessing fails or for debugging
            print("Falling back to serial A* computation...")
            solution_serial = {}
            for agent_name_s in self.agent_dict.keys():
                local_path_s = self.a_star.search(agent_name_s) # Original AStar search
                if not local_path_s:
                    return None
                solution_serial[agent_name_s] = local_path_s
            return solution_serial

    def compute_solution_cost(self, solution_dict):
        if not solution_dict: return float('inf')
        # Sum of individual path lengths (number of moves, so len(path)-1)
        # Or sum of makespans (time agent reaches goal, which is len(path)-1 if goal time is arrival time)
        # Standard SIC is sum of path lengths (number of states in path).
        cost = 0
        for path in solution_dict.values():
            if path: # Path could be None if compute_solution_for_node failed but wasn't caught
                cost += (len(path) -1) # Or len(path) if cost is sum of states occupied
            else: # Should not happen if solution_dict is valid
                return float('inf')
        return cost


class AStar():
    def __init__(self, env_ref):
        self.env = env_ref # Reference to the Environment instance

    def reconstruct_path(self, came_from, current_state):
        path = [current_state]
        while current_state in came_from:
            current_state = came_from[current_state]
            path.append(current_state)
        return path[::-1] # Return reversed path [start_state, ..., goal_state]

    def search(self, agent_name_to_plan_for):
        initial_state = self.env.agent_dict[agent_name_to_plan_for]["start"]
        
        # Max time for A* can be related to grid size or a fixed large number
        # For a grid WxH, a simple path is at most W*H. With time, it can be larger.
        # This helps prune branches that involve an agent waiting excessively long.
        max_planning_time = self.env.dimension[0] * self.env.dimension[1] + len(self.env.agent_dict) * 5 # Heuristic
        if self.env.dimension[0] * self.env.dimension[1] ==0: max_planning_time = 200 # for small/test grids

        # (f_score, tie_breaker_time, state_object)
        # Tie_breaker_time to prefer paths that reach goal sooner if f_scores are equal.
        # Or use state's __lt__ if State objects are directly comparable.
        open_pq = []
        # Push (f_score, state)
        initial_f_score = self.env.heuristic(initial_state, agent_name_to_plan_for)
        heapq.heappush(open_pq, (initial_f_score, initial_state))

        came_from = {} # child_state -> parent_state
        g_score = {initial_state: 0} # state -> cost_from_start

        expanded_count = 0

        while open_pq:
            expanded_count +=1
            if expanded_count > 20000 : # Safety break for A*
                # print(f"A* for {agent_name_to_plan_for} exceeded expansion limit")
                return False

            current_f, current_state = heapq.heappop(open_pq)

            # If we already found a shorter path to current_state, skip
            if current_f > g_score.get(current_state, -1) + self.env.heuristic(current_state, agent_name_to_plan_for) \
                and g_score.get(current_state, -1) != -1 : # Check if g_score was set
                continue
            
            if self.env.is_at_goal(current_state, agent_name_to_plan_for):
                 # Agent can wait at goal if other agents are still moving.
                 # Path length up to this point is g_score[current_state].
                 # The CBS cost is sum of path lengths (or makespans).
                 # For now, once goal location is reached, path is found.
                return self.reconstruct_path(came_from, current_state)

            if current_state.time >= max_planning_time: # Prune if path is getting too long
                if not self.env.is_at_goal(current_state, agent_name_to_plan_for):
                    continue

            # Pass agent_name for context-specific neighbor validation
            for neighbor_state in self.env.get_neighbors(current_state, agent_name_to_plan_for):
                tentative_g = g_score.get(current_state, float('inf')) + 1 # step_cost = 1

                if tentative_g < g_score.get(neighbor_state, float('inf')):
                    came_from[neighbor_state] = current_state
                    g_score[neighbor_state] = tentative_g
                    new_f_score = tentative_g + self.env.heuristic(neighbor_state, agent_name_to_plan_for)
                    heapq.heappush(open_pq, (new_f_score, neighbor_state))
        return False # No path found


class HighLevelNode(object):
    def __init__(self):
        self.solution = {} # agent_name -> [State, ...] path
        self.constraint_dict = {} # agent_name -> Constraints object
        self.cost = 0 # Sum of individual path lengths (or makespans)
        self._hash_val = None # Cache for hash

    def _get_hashable_constraints_representation(self):
        # Creates a canonical, hashable representation of the constraints
        sorted_constraints = []
        for agent_name in sorted(self.constraint_dict.keys()):
            constraints_obj = self.constraint_dict[agent_name]
            # Sort constraints for consistent hashing
            # For VertexConstraint: (time, x, y)
            # For EdgeConstraint: (time, x1, y1, x2, y2)
            vc_tuples = tuple(sorted([(vc.time, vc.location.x, vc.location.y) for vc in constraints_obj.vertex_constraints]))
            ec_tuples = tuple(sorted([(ec.time, ec.location_1.x, ec.location_1.y, ec.location_2.x, ec.location_2.y) for ec in constraints_obj.edge_constraints]))
            sorted_constraints.append((agent_name, vc_tuples, ec_tuples))
        return tuple(sorted_constraints)

    def __eq__(self, other):
        if not isinstance(other, HighLevelNode):
            return NotImplemented
        # Nodes are equal if their cost and their (canonical) constraints are identical.
        # Solution is derived, so primary comparison is on constraints and cost.
        if self.cost != other.cost:
            return False
        return self._get_hashable_constraints_representation() == \
               other._get_hashable_constraints_representation()

    def __hash__(self):
        if self._hash_val is None:
            self._hash_val = hash((self.cost, self._get_hashable_constraints_representation()))
        return self_hash_val

    def __lt__(self, other): # For heapq comparison
        if not isinstance(other, HighLevelNode):
            return NotImplemented
        if self.cost != other.cost:
            return self.cost < other.cost
        # Tie-breaking: prefer nodes with fewer total constraints as a heuristic (optional)
        # This makes the sort order consistent if costs are equal.
        self_num_constraints = sum(len(c.vertex_constraints) + len(c.edge_constraints) for c in self.constraint_dict.values())
        other_num_constraints = sum(len(c.vertex_constraints) + len(c.edge_constraints) for c in other.constraint_dict.values())
        if self_num_constraints != other_num_constraints:
            return self_num_constraints < other_num_constraints
        # Further tie-breaking if needed, e.g., by hash of constraints for determinism
        return hash(self._get_hashable_constraints_representation()) < hash(other._get_hashable_constraints_representation())


class CBS(object):
    def __init__(self, environment):
        self.env = environment # Environment object
        self.open_pq = [] # Min-priority queue for HighLevelNodes: (cost, node_object)
        self.closed_set = set() # Set of expanded HighLevelNode._get_hashable_constraints_representation() + cost
                                # to avoid re-expanding nodes with same constraints and cost.

    def search(self, max_iterations=50000): # Added max_iterations
        # 1. Create the root HighLevelNode
        root_node = HighLevelNode()
        # Initialize with empty constraints for all agents
        for agent_name in self.env.agent_dict.keys():
            root_node.constraint_dict[agent_name] = Constraints()

        # Compute initial solution for the root node
        root_solution = self.env.compute_solution_for_node(root_node.constraint_dict)
        if root_solution is None:
            print("CBS Error: No initial solution found for the root node.")
            return None 
        
        root_node.solution = root_solution
        root_node.cost = self.env.compute_solution_cost(root_node.solution)

        heapq.heappush(self.open_pq, root_node) # Push the node itself, __lt__ will be used
        
        iterations = 0
        while self.open_pq:
            iterations += 1
            if iterations > max_iterations:
                print(f"CBS: Exceeded max iterations ({max_iterations}). No solution found.")
                return None

            # 2. Pop the node P with the lowest cost from OPEN
            current_node = heapq.heappop(self.open_pq)

            # 3. Check if P has been closed (based on its essential properties)
            # We store a canonical representation of (cost, constraints) in closed_set
            node_signature = (current_node.cost, current_node._get_hashable_constraints_representation())
            if node_signature in self.closed_set:
                continue
            self.closed_set.add(node_signature)

            # 4. Validate P's solution to find the first conflict
            # The environment needs to know the constraints of the current_node for validation.
            self.env.current_hl_node_constraints = current_node.constraint_dict # Set context for env
            first_conflict = self.env.get_first_conflict(current_node.solution)

            # 5. If no conflict, solution found
            if first_conflict is None:
                print(f"CBS: Solution found in {iterations} iterations with cost {current_node.cost}.")
                return self.generate_plan_output(current_node.solution)

            # 6. If conflict exists, create new constraints
            newly_added_constraints_map = self.env.create_constraints_from_conflict(first_conflict)

            # 7. For each agent involved in the conflict, create a new child node
            for agent_to_constrain, specific_new_constraint in newly_added_constraints_map.items():
                child_node = HighLevelNode()
                child_node.constraint_dict = deepcopy(current_node.constraint_dict) # Inherit parent's constraints
                child_node.constraint_dict[agent_to_constrain].add_constraint(specific_new_constraint)
                child_node._hash_val = None # Must reset hash after modification

                # Re-plan for this child node with the new constraints
                self.env.current_hl_node_constraints = child_node.constraint_dict # Set context for env
                child_solution = self.env.compute_solution_for_node(child_node.constraint_dict)

                if child_solution is not None:
                    child_node.solution = child_solution
                    child_node.cost = self.env.compute_solution_cost(child_solution)
                    
                    child_signature = (child_node.cost, child_node._get_hashable_constraints_representation())
                    if child_signature not in self.closed_set: # Check against closed set before adding
                        heapq.heappush(self.open_pq, child_node)
        
        print(f"CBS: No solution found after {iterations} iterations (OPEN set is empty).")
        return None

    def generate_plan_output(self, solution_dict): # solution_dict: agent_name -> [State, ...]
        # Output format: a dictionary: {agent_name: list_of_coord_tuples}
        # Or: list of lists, if downstream requires [[path_agent0_coords], [path_agent1_coords], ...]
        output_plan = {}
        for agent_name in sorted(solution_dict.keys()): # Sort for consistent output if needed
            path_states = solution_dict[agent_name]
            coord_path = []
            if path_states: # Ensure path is not empty
                for state_obj in path_states:
                    coord_path.append((state_obj.location.x, state_obj.location.y))
            output_plan[agent_name] = coord_path
        return output_plan


# ... (其他類的定義保持不變) ...

def convert_to_grid_coordinates(image_width_px, image_height_px,
                                obstacle_center_coords_px, # List of (cx_px, cy_px)
                                grid_cell_size_px,
                                agent_start_goal_pairs_px,
                                global_obstacle_radius_px): # Global radius for all obstacles
    """
    Converts pixel-based input to grid-based representation for CBS.
    Args:
        image_width_px: Total width of the area in pixels.
        image_height_px: Total height of the area in pixels.
        obstacle_center_coords_px: List of (ox_px, oy_px) for obstacle centers.
        grid_cell_size_px: Size of one grid cell in pixels.
        agent_start_goal_pairs_px: List of ((start_x_px, start_y_px), (goal_x_px, goal_y_px)).
        global_obstacle_radius_px: Radius of all obstacles in pixels.
    Returns:
        dimension (tuple): (num_cols, num_rows) of the grid.
        agents_data (list): List of agent dicts {"name": str, "start": [gx, gy], "goal": [gx, gy]}.
        obstacles_grid (set): Set of (gx, gy) tuples representing obstacle cells.
    """
    num_cols = (image_width_px + grid_cell_size_px - 1) // grid_cell_size_px
    num_rows = (image_height_px + grid_cell_size_px - 1) // grid_cell_size_px
    grid_dimension = (num_cols, num_rows)

    agents_grid_data = []
    for i, (start_px, goal_px) in enumerate(agent_start_goal_pairs_px):
        start_gx = max(0, min(start_px[0] // grid_cell_size_px, num_cols - 1))
        start_gy = max(0, min(start_px[1] // grid_cell_size_px, num_rows - 1))
        goal_gx = max(0, min(goal_px[0] // grid_cell_size_px, num_cols - 1))
        goal_gy = max(0, min(goal_px[1] // grid_cell_size_px, num_rows - 1))
        agents_grid_data.append({
            "name": f"agent{i}",
            "start": [start_gx, start_gy],
            "goal": [goal_gx, goal_gy]
        })

    grid_obstacles = set()
    # obstacle_center_coords_px is a list of (center_x_px, center_y_px)
    for obs_center_x_px, obs_center_y_px in obstacle_center_coords_px: # Correctly unpack (cx, cy)
        min_gx_candidate = (obs_center_x_px - global_obstacle_radius_px) // grid_cell_size_px
        max_gx_candidate = (obs_center_x_px + global_obstacle_radius_px) // grid_cell_size_px
        min_gy_candidate = (obs_center_y_px - global_obstacle_radius_px) // grid_cell_size_px
        max_gy_candidate = (obs_center_y_px + global_obstacle_radius_px) // grid_cell_size_px

        for gx in range(max(0, int(min_gx_candidate)), min(num_cols, int(max_gx_candidate) + 1)):
            for gy in range(max(0, int(min_gy_candidate)), min(num_rows, int(max_gy_candidate) + 1)):
                cell_center_x_px = gx * grid_cell_size_px + grid_cell_size_px / 2.0
                cell_center_y_px = gy * grid_cell_size_px + grid_cell_size_px / 2.0

                dist_sq = (cell_center_x_px - obs_center_x_px)**2 + \
                          (cell_center_y_px - obs_center_y_px)**2
                if dist_sq <= (global_obstacle_radius_px)**2:
                    grid_obstacles.add((gx, gy))

    return grid_dimension, agents_grid_data, grid_obstacles


def cbs_planning(matched_target_and_array_batch_pixels,
                 obstacle_center_coordinate_pixels, # Renamed for clarity
                 grid_size_pixels,
                 image_width_pixels,
                 image_height_pixels,
                 global_obstacle_radius_pixels, # Global radius passed here
                 max_cbs_iterations=50000):
    """
    Main function to set up and run CBS planning.
    Inputs are in pixel units and will be converted to grid units.
    """
    print(f"CBS Planning started. Grid Size: {grid_size_pixels}px, Image: {image_width_pixels}x{image_height_pixels}px, Obstacle Radius: {global_obstacle_radius_pixels}px")

    grid_dimension, agents_data_for_env, obstacles_for_env = convert_to_grid_coordinates(
        image_width_pixels, image_height_pixels,
        obstacle_center_coordinate_pixels, # Pass the list of (center_x, center_y)
        grid_size_pixels,
        matched_target_and_array_batch_pixels,
        global_obstacle_radius_pixels # Pass the global radius
    )

    if not agents_data_for_env:
        print("CBS Warning: No agents to plan for after conversion.")
        return None

    cbs_environment = Environment(grid_dimension, agents_data_for_env, obstacles_for_env)
    cbs_solver = CBS(cbs_environment)

    solution = cbs_solver.search(max_iterations=max_cbs_iterations)

    if solution is None:
        print("CBS Solution not found or error occurred.")
        return None

    # print(f"CBS Solution found. Cost (sum of path lengths): {cbs_environment.compute_solution_cost(solution_if_dict_of_states(solution, cbs_environment))}")
    return solution


if __name__ == "__main__":
    print("CBS Example Usage:")
    example_agents_pixels = [
        ((10, 10), (70, 10)),
        ((10, 70), (70, 70))
    ]
    # Obstacle coordinate list ONLY has coordinates, radius is fixed
    example_obstacles_center_pixels = [(40, 10), (40, 40)] # List of (center_x, center_y)

    example_grid_size = 20
    example_img_width = 100
    example_img_height = 100
    example_global_obs_radius = 10 # All obstacles have this radius

    solution_paths = cbs_planning(
        example_agents_pixels,
        example_obstacles_center_pixels, # Pass list of centers
        example_grid_size,
        example_img_width,
        example_img_height,
        example_global_obs_radius, # Pass the global radius
        max_cbs_iterations=1000
    )

    if solution_paths:
        print("\nCBS Planning Succeeded. Paths (grid coordinates):")
        for agent_name, path in sorted(solution_paths.items()):
            print(f"  {agent_name}: {path}")
    else:
        print("\nCBS Planning Failed or No Solution Found in example.")

    print("\n--- Test Case: Agents swapping ---")
    example_agents_swap_pixels = [
        ((1*example_grid_size+5, 1*example_grid_size+5), (3*example_grid_size+5, 1*example_grid_size+5)),
        ((3*example_grid_size+5, 1*example_grid_size+5), (1*example_grid_size+5, 1*example_grid_size+5))
    ]
    example_obstacles_swap_pixels = []

    solution_paths_swap = cbs_planning(
        example_agents_swap_pixels,
        example_obstacles_swap_pixels,
        example_grid_size,
        example_img_width,
        example_img_height,
        example_global_obs_radius, # Still need to pass radius, even if no obstacles
        max_cbs_iterations=1000
    )
    if solution_paths_swap:
        print("\nCBS Swap Test Succeeded. Paths (grid coordinates):")
        for agent_name, path in sorted(solution_paths_swap.items()):
            print(f"  {agent_name}: {path}")
    else:
        print("\nCBS Swap Test Failed or No Solution Found.")