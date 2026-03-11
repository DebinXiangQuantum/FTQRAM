from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from qiskit.circuit.library import RYGate, RZGate
import numpy as np
Decompose_CSWAP = False
def cswap(cir:QuantumCircuit,*q):
    if Decompose_CSWAP:
        cir.rz(1.9986164569854736,q[1])
        cir.rx(0.1859593391418457,q[1])
        cir.cz( q[0],q[1])
        cir.rx(3.1414453983306885,q[0])
        cir.rz(0.39461636543273926,q[1])
        cir.rx(1.570793867111206,q[1])
        cir.rz(-2.325270414352417,q[2])
        cir.rx(1.4996445178985596,q[2])
        cir.cz( q[1],q[2])
        cir.rz(-3.141577959060669,q[1])
        cir.rx(2.356193780899048,q[1])
        cir.cz( q[0],q[1])
        cir.rx(3.141592264175415,q[0])
        cir.rz(-3.1415770053863525,q[1])
        cir.rx(0.8390493392944336,q[1])
        cir.rz(1.6863858699798584,q[2])
        cir.rx(1.6270678043365479,q[2])
        cir.cz( q[1],q[2])
        cir.rz(-1.5705323219299316,q[1])
        cir.rx(1.5710699558258057,q[1])
        cir.rz(-2.3577821254730225,q[2])
        cir.rx(1.5707881450653076,q[2])
        cir.cz( q[1],q[2])
        cir.rz(2.3577845096588135,q[1])
        cir.rx(1.6270687580108643,q[1])
        cir.cz( q[0],q[1])
        cir.rz(0.785407304763794,q[0])
        cir.rz(-1.5144445896148682,q[1])
        cir.rx(0.7853965759277344,q[1])
        cir.rx(1.5707967281341553,q[2])
        cir.cz( q[1],q[2])
        cir.rx(0.7854020595550537,q[1])
        cir.rz(-1.6244618892669678,q[2])
        cir.rx(1.5710680484771729,q[2])
        cir.cz( q[1],q[2])
        cir.rz(-0.1719667911529541,q[1])
        cir.rx(1.4999163150787354,q[1])
        cir.rz(-0.8163588047027588,q[1])
        cir.rz(-1.4985880851745605,q[2])
        cir.rx(1.399273157119751,q[2])
        cir.rz(-2.39945387840271,q[2])
    else:
        cir.cswap(*q)

class RouterQubit:
    def __init__(self, index, level, direction, root):
        self.index = index
        self.level = level
        self.root = root
        self.left_router = None
        self.right_router = None
        self.direction = direction
        
        self.qreg = QuantumRegister(1,self.reg_name)

    @property
    def address(self):
        if self.root is None:
            return self.direction
        else:
            return self.root.address + self.direction

    @property
    def reg_name(self):
        if self.address:
            return f"router_{self.level}_{self.address}"
        else:
            return f"router_{self.level}"

    def add_leaf_qubits(self, circuit):
        self.left =  QuantumRegister(1,self.reg_name + '_l')
        self.right =  QuantumRegister(1,self.reg_name + '_r')
        circuit.add_register(self.left)
        circuit.add_register(self.right)
    
    def add_data_qubits(self, circuit):
        self.data =  QuantumRegister(1,self.reg_name + '_data')
        circuit.add_register(self.data)
        
class Qram:
    def __init__(self, address, data,*regs, bandwidth=1):
        self.address = address
        self.data = data
        self.bandwidth = bandwidth
        self.apply_classical_bit = True
        self.circuit = QuantumCircuit(*regs)
        self.routers = [[] for _ in range(len(address[0]) if isinstance(address, list) else address)]
    def add_router_tree(self, level,root):
        self.circuit.add_register(root.qreg)
        if root.left_router is not None:
            self.circuit.add_register(root.left_router.qreg)
        if root.right_router is not None:
            self.circuit.add_register(root.right_router.qreg)
        self.routers[level].append(root)
        if root.left is not None:
            self.add_router_tree(level + 1, root.left)
        if root.right is not None:
            self.add_router_tree(level + 1, root.right)

    def add_incident_qubits(self, incident):
        self.incident = incident
        self.circuit.add_register(self.incident.qreg)

    def __call__(self, *args):
        if len(args) == 3 and isinstance(args[0], QuantumCircuit):
            circuit, address_qubits, bus_qubits = args
            self.circuit = circuit
        elif len(args) == 2:
            address_qubits, bus_qubits = args
            circuit = self.circuit
        else:
            raise ValueError("Expected (circuit, address_qubits, bus_qubits) or (address_qubits, bus_qubits)")
        self.address_qubits = address_qubits
        self.bus_qubits = bus_qubits
        self.n_addr = getattr(address_qubits, '_size', len(address_qubits))
        self.decompose_circuit(circuit)

    def router(self, circuit: QuantumCircuit, router: QuantumRegister, incident, left, right):
        circuit.x(router)
        cswap(circuit,router, incident, left)
        circuit.x(router)
        cswap(circuit,router, incident, right)

    def reverse_router(self, circuit: QuantumCircuit, router, incident, left, right):
        cswap(circuit,router, incident, right)
        circuit.x(router)
        cswap(circuit,router, incident, left)
        circuit.x(router)

    def layers_router(self, circuit, router_obj, incident, address_index, mid):
        if router_obj.level == 0:
            circuit.swap(incident, mid.qreg)
        if router_obj.level == 0 and address_index == 0:
            circuit.swap(router_obj.qreg, mid.qreg)
        else:
            if router_obj.level + 2 == address_index and address_index == self.n_addr:
                self.router(circuit, router_obj.qreg, mid.qreg, router_obj.left.qreg, router_obj.right.qreg)
                return
            
            self.router(circuit, router_obj.qreg, mid.qreg, router_obj.left_router.qreg, router_obj.right_router.qreg)
            if router_obj.level + 1 == address_index:
                if router_obj.left_router.qreg != router_obj.left:
                    circuit.swap(router_obj.left_router.qreg, router_obj.left.qreg)
                if router_obj.right_router.qreg != router_obj.right:
                    circuit.swap(router_obj.right_router.qreg, router_obj.right.qreg)
                return
            if router_obj.left is not None:
                self.layers_router(circuit, router_obj.left, router_obj.left_router, address_index, router_obj.left_router)
            if router_obj.right is not None:
                self.layers_router(circuit, router_obj.right,router_obj.right_router,  address_index, router_obj.right_router)

    def reverse_layers_router(self, circuit, router_obj, incident, address_index, mid):
        if address_index != 0:
            if router_obj.level + 1 > address_index:
                return
            if router_obj.level + 2 == address_index and address_index == self.n_addr:
                self.reverse_router(circuit, router_obj.qreg, mid.qreg, router_obj.left.qreg, router_obj.right.qreg)
            else:
                if router_obj.right is not None:
                    self.reverse_layers_router(circuit, router_obj.right, router_obj.right_router, address_index, router_obj.right_router)
                if router_obj.left is not None:
                    self.reverse_layers_router(circuit, router_obj.left, router_obj.left_router, address_index, router_obj.left_router)
                if router_obj.level + 1 == address_index and address_index != self.n_addr:
                    if router_obj.left_router.qreg != router_obj.left:
                        circuit.swap(router_obj.left_router.qreg, router_obj.left.qreg)
                    if router_obj.right_router.qreg != router_obj.right:
                        circuit.swap(router_obj.right_router.qreg, router_obj.right.qreg)
                if router_obj.right_router is not None:
                    self.reverse_router(circuit, router_obj.qreg, mid.qreg, router_obj.left_router.qreg, router_obj.right_router.qreg)
        else:
            circuit.swap(router_obj.qreg, mid.qreg)
        if router_obj.level == 0:
            circuit.swap(incident, mid.qreg)

    def _leaf_cz(self, circuit, mid, leaf, leaf_address):
        """Apply CZ interaction between mid (displaced last addr bit) and leaf (bus).

        leaf_address is the tree prefix for this leaf. The full data index is
        leaf_address + last_addr_bit.
        """
        # mid=|1⟩ case: data[leaf_addr + "1"]
        if self.data[int(leaf_address + '1', 2)] == 1:
            circuit.cz(mid.qreg, leaf.qreg)
        # mid=|0⟩ case: data[leaf_addr + "0"]
        if self.data[int(leaf_address + '0', 2)] == 1:
            circuit.x(mid.qreg)
            circuit.cz(mid.qreg, leaf.qreg)
            circuit.x(mid.qreg)

    def bus_route_interact_unroute(self, circuit, router_obj, mid, address_index):
        """Route bus to leaves, apply CZ interaction, then reverse route.

        At the deepest level, mid holds the displaced last address bit and
        the leaf routers hold the bus. The CZ is applied between mid and each leaf.
        """
        if router_obj.level + 1 > address_index:
            return
        if router_obj.level + 2 == address_index and address_index == self.n_addr:
            # Forward: route to leaves
            self.router(circuit, router_obj.qreg, mid.qreg, router_obj.left.qreg, router_obj.right.qreg)
            # CZ interaction on each leaf using the child's address
            self._leaf_cz(circuit, mid, router_obj.left, router_obj.left.address)
            self._leaf_cz(circuit, mid, router_obj.right, router_obj.right.address)
            # Reverse
            self.reverse_router(circuit, router_obj.qreg, mid.qreg, router_obj.left.qreg, router_obj.right.qreg)
        else:
            # Forward: route through intermediate nodes
            self.router(circuit, router_obj.qreg, mid.qreg, router_obj.left_router.qreg, router_obj.right_router.qreg)
            # Recurse into children
            if router_obj.left is not None:
                self.bus_route_interact_unroute(circuit, router_obj.left, router_obj.left_router, address_index)
            if router_obj.right is not None:
                self.bus_route_interact_unroute(circuit, router_obj.right, router_obj.right_router, address_index)
            # Reverse
            self.reverse_router(circuit, router_obj.qreg, mid.qreg, router_obj.left_router.qreg, router_obj.right_router.qreg)

    def decompose_circuit(self, circuit):
        for idx in self.bus_qubits:
            circuit.h(idx)
        n_bus = getattr(self.bus_qubits, '_size', len(self.bus_qubits))
        incidents = {i:self.address_qubits[i] for i in range(self.n_addr)}
        incidents.update({i+self.n_addr:self.bus_qubits[i] for i in range(n_bus)})
        # Route address bits into the tree
        for idx in range(self.n_addr):
            self.layers_router(circuit, self.routers[0][0], incidents[idx], idx, self.incident)
            circuit.barrier()
        # Route bus, interact at leaves, and unroute bus
        for b_idx in range(n_bus):
            bus_incident = incidents[self.n_addr + b_idx]
            circuit.swap(bus_incident, self.incident.qreg)
            self.bus_route_interact_unroute(circuit, self.routers[0][0], self.incident, self.n_addr)
            circuit.swap(bus_incident, self.incident.qreg)
            circuit.barrier()
        # Reverse route address bits out of the tree
        for idx in reversed(range(self.n_addr)):
            self.reverse_layers_router(circuit, self.routers[0][0], incidents[idx], idx, self.incident)
            circuit.barrier()
        for idx in self.bus_qubits:
            circuit.h(idx)
def generate_grid_coupling_map(width, height):
    coupling_map = []
    for i in range(width):
        for j in range(height):
            if i < width - 1:
                coupling_map.append([i*height+j, (i+1)*height+j])
            if j < height - 1:
                coupling_map.append([i*height+j, i*height+j+1])
    return coupling_map

def plot_coupling_map(coupling_map, width, height):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(width, height))
    for i in range(width):
        for j in range(height):
            plt.text(i, j, str(i*height+j), ha='center', va='center', fontsize=12)
            for edge in coupling_map:
                plt.plot([edge[0]%height, edge[1]%height], [edge[0]//height, edge[1]//height], 'k-')
    plt.axis('off')
    plt.show()

def cz_depth(circuit):
    new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    for gate in circuit.data:
        if gate[0].name == 'cz':
            new_circuit.data.append(gate)
    
    return new_circuit.depth()
            
if __name__ == "__main__":
    address_bits = 3
    address = [bin(i)[2:].zfill(address_bits) for i in range(2**address_bits)]
    data = [0,0,1,1,1,0,0,1]
    address_qregs = QuantumRegister(address_bits, 'address')
    bus_qregs = QuantumRegister(1, 'bus')
    bus_cregs = ClassicalRegister(1, 'bus_classical')
    address_cregs = ClassicalRegister(address_bits, 'address_classical')

    qram = Qram(address, data, address_qregs, bus_qregs, address_cregs, bus_cregs, bandwidth=1)
    circuit = qram.circuit

    # Build router tree
    counters = {level: 0 for level in range(address_bits)}
    def make_router(level, direction, parent):
        idx = counters[level]
        counters[level] += 1
        node = RouterQubit(idx, level, direction, parent)
        node.left_router = RouterQubit(0, level, 'l', node)
        node.right_router = RouterQubit(0, level, 'r', node)
        if level < address_bits - 1:
            node.left = make_router(level + 1, '0', node)
            node.right = make_router(level + 1, '1', node)
        else:
            node.left = None
            node.right = None
        return node

    root = make_router(0, '', None)
    incident = RouterQubit(0, 0, 'inc', None)
    qram.add_router_tree(0, root)
    qram.add_incident_qubits(incident)

    for i in range(address_bits):
        circuit.h(address_qregs[i])

    qram(address_qregs, bus_qregs)
    circuit.measure(bus_qregs, bus_cregs)
    circuit.measure(address_qregs, address_cregs)
    print(circuit.num_qubits)
    simulator = Aer.get_backend('qasm_simulator')
    result = simulator.run(circuit, shots=10000).result()
    counts = result.get_counts(circuit)
    print("测量结果：", {k[::-1]: v/10000 for k, v in counts.items()})
