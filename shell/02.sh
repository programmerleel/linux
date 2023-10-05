#!/bin/sh

function func_1(){
    var_1=100
}

func_1

echo $var_1

#func_2执行结束，var_2释放

function func_2(){
    local var_2=200
    local var_1=200
}

func_2

echo $var_1
echo $var_2

#全局变量在一个shell进程中的所有执行的sh脚本中有效

var_a=1

echo $var_a