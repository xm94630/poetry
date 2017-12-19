# encoding=utf-8
import numpy as np
import tensorflow as tf


# 我认为是变量相加的案例
def task1():
    """
    使用tensorlfow，你首先要创建一个图，如这里的a、b（这里简化为一个整数，复杂点的可以是数组、矩阵）
    然后通过会话来流动这张图
    """
    a = tf.constant(111)
    b = tf.constant(222)
    product = tf.add(a, b) #相加
    result = tf.Session().run(product)
    print(result)


# 这里是state保存每个状态的案例
def task2():
    """
    这个方法示例'Variable'的作用，它相当于一个存储器，存储中间变量
    """
    a = tf.constant(10)
    # 首先我们定义一个Variable，名字叫state，初始值是38
    state = tf.Variable(10000, name='state')
    new_value = tf.add(state, a) #相加
    update = tf.assign(state, new_value)

    # 使用variable，在用会话启动它之前要初始化一下'Variable'
    # 要不然tf怎么知道你设定的初始值是多少呢？
    init_op = tf.initialize_all_variables()

    # 注意这里 with as的用法
    # 这里是 state 和 变量a 相加的结果输出，本身 state 没有得到更新
    with tf.Session() as sess:
        sess.run(init_op) #这个返回none
        print(sess.run(new_value))

    # 这里是 state 和 变量a 相加后，本身 state 得到更新，最后记录的是 state 的状态
    with tf.Session() as sess:
        sess.run(init_op)
        #这里循环3次，分别为0，1，2
        for i in range(3):
            # 执行这一步把state和a相加的值，得加到state自身
            sess.run(update)
            # 每一步执行之后我们看看state的值
            print(sess.run(state))

# 这个例子是对预值的处理
def task3():
    """
    Feed data进入图之中，入口是placeholder，相当于占位符先把入口霸占一下，
    等数据来了再从这里进入图之中
    """
    x = tf.placeholder(tf.float32, None, name='aaa')
    y = tf.placeholder(tf.float32, None, name='bbb')
    a = 1
    b = 2
    # 我们指定了两个数据流入的入口，并且固定了形状(数据类型)，如果输入不对会报错
    with tf.Session() as sess:
        result = sess.run(tf.add(x, y), feed_dict={x: a, y: b})
        print(result)

if __name__ == '__main__':
    task1();
    #task2();
    #task3();



