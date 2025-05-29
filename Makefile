CXX = nvcc

TAU?=1.0
MONITOR?=1000

INCLUDES = -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/math_libs/12.2/include 
FLAGS = --expt-extended-lambda -lcufft -std=c++17 -arch=sm_75 #-DSPECTRALCN
PARAMSEW = -DC2=1.0 -DTAU=$(TAU) -DMONITOR=$(MONITOR) #-DDOUBLE  
PARAMSKPZ = -DC2=1.0 -DKPZ=1.0 -DTAU=$(TAU) -DMONITOR=$(MONITOR)  #-DDOUBLE  
PARAMSANH = -DC2=1.0 -C4=1.0 -DTAU=$(TAU) -DMONITOR=$(MONITOR) #-DDOUBLE  

LDFLAGS = -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/math_libs/12.2/lib64 


activeinterface: ew.cu
	$(CXX) $(FLAGS) $(PARAMSEW) ew.cu -o activeinterface $(LDFLAGS) $(INCLUDES) 

anhinterface: ew.cu
	$(CXX) $(FLAGS) $(PARAMSANH) ew.cu -o anhinterface $(LDFLAGS) $(INCLUDES) 

kpzinterface: ew.cu
	$(CXX) $(FLAGS) $(PARAMSKPZ) ew.cu -o kpzinterface $(LDFLAGS) $(INCLUDES) 

update_git:
	git add *.cu Makefile *.h *.sh README.md ; git commit -m "program update"; git push

clean:
	rm activeinterface kpzinterface anhinterface
