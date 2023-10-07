#!/bin/sh

a=1
b=2
c=3
d=4

if [ $a -ne $b ]
then
    echo 'a=b'
fi
if [ $a -lt $b ]
then
    echo 'a<b'
elif [ $d -ge $c ]
then
    echo 'd>=c'
else
    echo 'else'
fi

if [ $a -lt $b ]&&[ $c -le $d ]
then
    echo '&&'
fi

if [ $a -gt $b ]||[ $c -ge $d ]
then
    echo '||'
fi

case $a in
1)
    echo 1
;;
2)
    echo 2
;;
*)
    echo '*'
;;
esac

for(( i=0;i<10;i++ ))
do
    echo $i
done

for i in $a $b $c $d
do
    echo $i
done

echo $0
$1
echo $1
$2
echo $2
$3
echo $3

echo $#

echo $*
echo $@

for i in "$*"
do
    echo $i
done

for i in "$@"
do
    echo $i
done

while [ $a -lt $d ]
do
    echo $a
    a=$[$a+1]
done