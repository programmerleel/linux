# linux

### linux访问github.com

- 访问www.ip33.com/dns.html查询github的DNS
- 修改/etc/hosts文件 添加查询到的dns以及github.com
- 运行/etc/init.d/nscd restart重启刷新DNS（可能会出现无法找到文件 需要安装nscd）
