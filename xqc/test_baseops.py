import jax.numpy as jnp
from xqc.baseops import Op, sx, sy, sz, id2, tensor, comm, acomm, tr, ptr, su

# Test multiplication
op = Op(jnp.eye(2))
op2 = op * 2
print('mul result:', op2.operator)

# Test addition with matching subs
op3 = Op(jnp.array([[0,1],[1,0]]))
op_sum = op + op3
print('add result:', op_sum.operator)

# Test addition with mismatched subs
op4 = Op(jnp.eye(4), subs=jnp.array([[2,2],[2,2]]))
try:
    op + op4
except Exception as e:
    print('add mismatched subs error:', e)

# Test dag
dag_op = op.dag()
print('dag subs:', dag_op.subs)

# Test sz
sz_op = sz()
print('sz operator:', sz_op.operator)

# Test ptr: create 2-qubit system
op_a = Op(jnp.kron(jnp.array([[1,0],[0,0]]), jnp.eye(2)), subs=jnp.array([[2,2],[2,2]]))
# keep subsystem 0
ptr_op = ptr(op_a, jnp.array([0]))
print('ptr shape:', ptr_op.operator.shape)

# Test su
basis = su(2)
print('su basis shape:', basis.shape)
