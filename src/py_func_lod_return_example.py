import paddle.fluid as fluid
import numpy as np


# 定义运算场所
place = fluid.CPUPlace()
# 创建执行器
exe = fluid.Executor(place)


def create_tmp_var(name, dtype, shape, lod_leval=0):
    return fluid.default_main_program().current_block().create_var(name=name,
                                                                   dtype=dtype,
                                                                   shape=shape,
                                                                   lod_leval=lod_leval)


def produce_tensor():
    print("create a tensor start")
    data0 = np.array([[[1, 2, 3, 4], [2, 3, 4, 5]], [[3, 4, 5, 6]]])
    data1 = np.array([[[3, 4, 5, 6]]])
    data = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [3, 4, 5, 6]])
    res = fluid.LoDTensor()
    res.set(data, place)
    res.set_lod([[0, 2, 3], [0, 2, 3, 4]])
    print("create a tensor end")
    return res


def print_func(var):
    print("in py func type: {0}".format(type(var)))
    print("in py func shape: {0}".format(var.shape()))
    print("in py func lod: {0}".format(var.lod()))
    print(np.array(var))


shape = [4]
out = create_tmp_var(None, dtype=np.int32, shape=shape, lod_leval=2)
fluid.layers.py_func(func=produce_tensor, x=None, out=out)
fluid.layers.py_func(func=print_func, x=out, out=None)
exe.run(fluid.default_startup_program())
oo = exe.run(fluid.default_main_program(), fetch_list=[out], return_numpy=False)
print(np.array(oo[0]))