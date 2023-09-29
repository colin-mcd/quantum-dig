import qiskit as Q
import sys
import numpy as np

EPS = 1e-10

def log2(n):
    return n.bit_length() - 1

def circuit_to_dag(circ):
    return Q.converters.circuit_to_dag(circ)

def circuit_to_dep_graph(circ):
    return dag_to_dep_graph(circuit_to_dag(circ), lambda x: circ.find_bit(x)[0])

def reachable(dag, gai, gbi):
    frontier = {gai}
    checked = set()
    #print(gai, gbi)
    while frontier:
        gi = frontier.pop()
        checked.add(gi)
        if gi == gbi:
            return True
        else:
            for gj in dag._multi_graph.successors(gi):
                if gj._node_id not in checked:
                    frontier.add(gj._node_id)
    return False

def trim_edges(dag):
    def get_id(node):
        if '_node_id' in dir(node):
            return node._node_id
        elif 'node_id' in dir(node):
            return node.node_id

    todo = set()
    if isinstance(dag, Q.dagcircuit.DAGDependency):
        #print(dag.get_all_edges())
        todo = set((ga, gb) for ga, gb, _ in dag.get_all_edges())
    else:
        todo = set((get_id(ga), get_id(gb)) for (ga, gb, qi) in dag.edges())

    #edges = dag.get_all_edges() if isinstance(dag, Q.dagcircuit.DAGDependency) else dag.edges()
    #todo = set((ga._node_id, gb._node_id, qi) for (ga, gb, qi) in edges)
    while todo:
        edge = todo.pop()
        gai, gbi = edge
        dup_edges = 0
        for gci in dag._multi_graph.successors(gai):
            if get_id(gci) == gbi:
                dup_edges += 1
        if dup_edges > 1:
            print(f"Dup edges!: {dup_edges}")
        if any(reachable(dag, get_id(gci), gbi) and gbi != get_id(gci) for gci in dag._multi_graph.successors(gai)):
            dag._multi_graph.remove_edge(gai, gbi)

def is_op_node(node):
    return isinstance(node, Q.dagcircuit.DAGOpNode)

class EdgeGraph:
    
    def __init__(self, nodes=(), edges=()):
        self.nodes = list(nodes)
        self.adj_matrix = [[None for _ in nodes] for _ in nodes]
        for fm_idx, to_idx, w in edges:
            self.adj_matrix[fm_idx][to_idx] = w
    
    def get_edge(self, fm_idx, to_idx):
        return self.adj_matrix[fm_idx][to_idx]

    def __getitem__(self, key):
        fm_idx, to_idx = key
        return self.get_edge(fm_idx, to_idx)

    def __setitem__(self, key, val):
        fm_idx, to_idx = key
        self.adj_matrix[fm_idx][to_idx] = val

    def __iadd__(self, key, val):
        fm_idx, to_idx = key
        self.adj_matrix[fm_idx][to_idx] += val

    def __isub__(self, key, val):
        fm_idx, to_idx = key
        self.adj_matrix[fm_idx][to_idx] -= val

    def __delitem__(self, key):
        fm_idx, to_idx = key
        self.adj_matrix[fm_idx][to_idx] = None

    def __iter__(self):
        return ((fm_idx, to_idx, w) \
                for fm_idx, row in enumerate(self.adj_matrix) \
                for to_idx, w in enumerate(row) \
                if w)

    def in_edges(self, idx):
        return ((fm_idx, idx, row[idx]) \
                for fm_idx, row in enumerate(self.adj_matrix) \
                if row[idx])

    def out_edges(self, idx):
        return ((idx, to_idx, w) \
                for to_idx, w in enumerate(self.adj_matrix[idx]) \
                if w)

import rustworkx as rx
import cProfile

def dag_to_dep_graph(dag, find_bit_fn):
    graph = rx.PyDAG(multigraph=False)
    graph.add_nodes_from(dag._multi_graph.nodes())
    todo = set()
    for gai, gbi, qi in dag._multi_graph.weighted_edge_list():
        graph.add_edge(gai, gbi, None)
        if is_op_node(graph.get_node_data(gai)) and is_op_node(graph.get_node_data(gbi)):
            todo.add((gai, gbi))
    visited = set()

    def maybe_push_edge(ifm, ito, gfm, gto):
        """
        Pushes an edge to `todo` if we need to visit it
        Args:
          ifm = index of source gate
          ito = index of target gate
          gfm = source gate
          gto = target gate
        """

        opfm = is_op_node(gfm)
        opto = is_op_node(gto)
        
        if opfm and opto:
            shared = any(find_bit_fn(q1) == find_bit_fn(q2)
                         for q1 in gfm.qargs for q2 in gto.qargs)
            if shared and (ifm, ito) not in todo \
               and (ifm, ito) not in visited:
                graph.add_edge(ifm, ito, None)
                todo.add((ifm, ito))
        elif opfm:
            if any(find_bit_fn(q) == find_bit_fn(gto.wire) for q in gfm.qargs):
                graph.add_edge(ifm, ito, None)
        elif opto:
            if any(find_bit_fn(q) == find_bit_fn(gfm.wire) for q in gto.qargs):
                graph.add_edge(ifm, ito, None)
        else:
            if gfm.wire == gto.wire:
                graph.add_edge(ifm, ito, None)

    while todo:
        gai, gbi = todo.pop()
        visited.add((gai, gbi))
        ga = graph.get_node_data(gai)
        gb = graph.get_node_data(gbi)
        if commutes(ga, gb, find_bit_fn):
            graph.remove_edge(gai, gbi)
            # Propagate in-edges
            for pidx, _, _ in graph.in_edges(gai):
                maybe_push_edge(pidx, gbi, graph.get_node_data(pidx), gb)
            # Propagate out-edges
            for _, cidx, _ in graph.out_edges(gbi):
                maybe_push_edge(gai, cidx, ga, graph.get_node_data(cidx))
    dag._multi_graph = graph
    return dag

#def is_sq_pow2(mtx):
#    return len(mtx.shape) == 2 and \
#           mtx.shape[0] == mtx.shape[1] and \
#           mtx.shape[0].bit_count() == 1

def as_bits(n, nbits=None):
    "Converts an integer into its bit representation"
    return [int(bool(n & (1 << (i - 1)))) for i in range(nbits or n.bit_length(), 0, -1)]

def rearrange_gate2(mat, old, new):
    """
    Rearranges a gate's unitary matrix for application to a new set of qubits.
    Assumes set(old) = set(new).
    Then you can multiply a gate @ rearrange_gate(...) to get the rearranged unitary.
    """
    old = list(reversed(old))
    new = list(reversed(new))
    qubits = len(new)
    old_idx = {o:i for i, o in enumerate(old)}
    new_idx = {n:i for i, n in enumerate(new)}
    old2new_idx = {o:new_idx[o] for o in old}
    new2old_idx = {n:old_idx[n] for n in new}

    size = 1 << qubits
    I = np.identity(size, dtype=np.dtype('int64'))
    
    bit_map = np.array([I[new2old_idx[n]] for n in new])
    bit_mat = np.array([as_bits(i, qubits) for i in range(size)])
    mapped = bit_mat @ bit_map
    for i in range(qubits - 1):
        mapped[:, i] <<= qubits - i - 1
    reordered = mapped.sum(axis=1)

    mat1 = np.ndarray(mat.shape, dtype=mat.dtype)
    mat2 = np.ndarray(mat.shape, dtype=mat.dtype)
    
    for i in range(size):
        mat1[i, :] = mat[reordered[i], :]
    for i in range(size):
        mat2[:, i] = mat1[:, reordered[i]]
    return mat2

def align_gates(ga, gb, find_bit_fn):
    """
    Aligns gates along the same qubits, returning a tuple of their new unitaries
    """
    ma = ga.op.to_matrix()
    mb = gb.op.to_matrix()
    #qas = [qi.index for qi in ga.qargs]
    #qbs = [qi.index for qi in gb.qargs]
    qas = [find_bit_fn(qi) for qi in ga.qargs]
    qbs = [find_bit_fn(qi) for qi in gb.qargs]
    qas_ins = list(set(qbs) - set(qas))
    qbs_ins = list(set(qas) - set(qbs))
    qas2 = qas + qas_ins
    qbs2 = qbs + qbs_ins
    # Add additional qubits from gb
    ma2 = np.kron(np.identity(1 << len(qas_ins), dtype=ma.dtype), ma)
    # Add additional qubits from ga
    mb2 = np.kron(np.identity(1 << len(qbs_ins), dtype=mb.dtype), mb)
    # Rearrange mb2 to match ma2's qubit order
    mb3 = rearrange_gate2(mb2, qbs2, qas2)
    return (ma2, mb3)


def commutes(ga, gb, find_bit_fn):
    ma, mb = align_gates(ga, gb, find_bit_fn)
    return (np.abs((ma @ mb) - (mb @ ma)) < EPS).all()


def read_qasm(qasm_file):
    acc = []
    with open(qasm_file, 'r') as fh:
        for line in fh:
            if not line.startswith('//'):
                acc.append(line)
    return Q.QuantumCircuit.from_qasm_str(''.join(acc))

def glen(generator):
    return sum(1 for _ in generator)

import time

def op_edges(g):
    return glen(filter(lambda e: isinstance(e[0], Q.dagcircuit.DAGOpNode) and isinstance(e[1], Q.dagcircuit.DAGOpNode), g.edges()))

def main(argv):
    if len(argv) == 2:
        circuit = read_qasm(argv[1])
        my_dag = circuit_to_dag(circuit)
        qk_dag = circuit_to_dag(circuit)
        orig_depth = my_dag.depth()
        orig_edges = glen(my_dag.edges())
        print(f"Original DAG: {orig_depth - 1} depth, {orig_edges} edges")

        my_start = time.time()
        my_dep = dag_to_dep_graph(my_dag, lambda x: circuit.find_bit(x)[0])
        my_end = time.time()

        #trim_edges(my_dep)
        #untrimmed_edges = my_dep.edges()
        print(f"My dependency graph: {my_dep.depth() - 1} depth, {op_edges(my_dep)} edges, {my_end - my_start:0.4f} sec")
        return None
        # Possible buggy difference between dag2dagdep and circ2dagdep...?
        qk_start = time.time()
        qk_dep = Q.converters.dag_to_dagdependency(qk_dag)
        qk_end = time.time()

        #trim_edges(qk_dep)

        print(f"Qiskit dependency graph: {qk_dep.depth()} depth, {glen(qk_dep.get_all_edges())} edges, {qk_end - qk_start:0.4f} sec")
    else:
        print("Pass a .qasm file as arg", out=sys.stderr)

if __name__ == '__main__':
    cProfile.run('main(sys.argv)')
    #main(sys.argv)
