#!/bin/bash

# by author Lee

while :

	echo "input number(between 1 and 4):"
	read number
	echo "input number is ${number}"
	do
		case ${number} in
			1|2|3|4) echo "1|2|3|4"
			;;
			#*) echo "*" 
			#continue
			*) echo "*"
		       	break
			;;
		esac
	done
				
