# linux

### linux访问github.com（稳定性有待考证）

- 访问www.ip33.com/dns.html查询github的DNS
- 修改/etc/hosts文件 添加查询到的dns以及github.com
- 运行/etc/init.d/nscd restart重启刷新DNS（可能会出现无法找到文件 需要安装nscd）

### 博客（简单轻量化）

- 采用json文件保存相关数据，一次加载，访问速度快，简单且轻量化
- md文件采用editormd组件进行编辑，markdown-it组件进行渲染
- 采用flask+tornnado+nginx进行部署
