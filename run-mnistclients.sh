set root=C:\Users\john.doe\AppData\Local\Continuum\anaconda3

call %root%\Scripts\activate.bat %root%
#!/bin/bash

#python server.py &
#sleep 3 # Sleep for 2s to give the server enough time to start

for i in `seq 0 9`; do
    echo "Starting client $i"
    python client-MNIST.py --seed=$i &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait

trap 'sleep infinity' EXIT
