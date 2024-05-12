#!/bsn/bash

# by author Lee
echo "input number(between 1 and 4):"
read number
echo "input number is ${number}"

case $number in
    1) echo "1"
    ;;
    2) echo "2"
    ;;
    3) echo "3"
    ;;
    4) echo "4"
    ;;
    *) echo "*"
    ;;
esac


