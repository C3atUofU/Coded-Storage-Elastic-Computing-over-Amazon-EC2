#!/bin/sh
KEY_PAIR="-i /home/nwoolsey/.ssh/MyKeyPair_1.pem"
for i in $(seq 1 2)
do
	echo $NODE
	NODE="node$i"
	ssh $KEY_PAIR ec2-user@${NODE} "echo 1"
        wait
done
for i in $(seq 1 2)
do
	(NODE="node$i"
	 USER_NODE="ec2-user@${NODE}"
	 SSH_MSG="ssh $KEY_PAIR $USER_NODE"
	 $SSH_MSG "sudo yum update -y"
	 $SSH_MSG "sudo yum install openmpi-devel -y"
	 FILE="bash_lines.txt"
	 if $SSH_MSG "test -f "$FILE""; then
	     echo "$FILE exist"
	 else
	     echo "$FILE not exist"
	     scp $KEY_PAIR $FILE ${USER_NODE}:$FILE
	     $SSH_MSG "sudo cat bash_lines.txt >> .bashrc"
	 fi
	 $SSH_MSG "sudo yum install tc -y"
	 $SSH_MSG "sudo yum install python-pip -y"
	 $SSH_MSG "sudo pip install tcconfig"
	 $SSH_MSG "pip install mpi4py --user"
	 $SSH_MSG "pip install numpy --user"
	 scp $KEY_PAIR mpi_hello_world.c ${USER_NODE}:mpi_hello_world.c
	 $SSH_MSG "mpicc mpi_hello_world.c -o mpi_hello_world.x"
	 FILE="master_pub_key"
	 if $SSH_MSG "test -f "$FILE""; then
	     echo "$FILE exist"
	 else
	     echo "$FILE not exist"
	     scp $KEY_PAIR master_pub_key ${USER_NODE}:master_pub_key
	     $SSH_MSG "cat master_pub_key >> ~/.ssh/authorized_keys"
         fi
	 ) &
done
wait
