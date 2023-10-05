# shell编程

#!/bin/sh：指定shell的解释器

source：直接在当前的shell中执行 
bash：开启子shell

- 环境变量：
  - export导出全局变量为环境变量
  - 对当前shell进程及其子进程有效
  - 环境变量传子不传父（子进程中修改，在父进程中不会改变）
- 全局变量：
  - shell在函数内创建的变量默认为全局变量，函数内外一致
  - 在一个shell进程中，所有sh脚本共享全局变量
- 局部变量：
  - 使用local进行定义
  - 仅在函数内部有效
- 只读变量：readonly
- 撤销变量：unset（readonly不可）