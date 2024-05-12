#!/bin/bash

# by author Lee

a=0
until [ ! $a -lt 5 ]
do
   echo $a
   a=`expr $a + 1`
done


