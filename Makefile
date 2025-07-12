CXX = nvcc

TAU?=1.0
MONITOR?=1000
Dt?=0.1

INCLUDES = -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/math_libs/12.2/include 
FLAGS = --expt-extended-lambda -lcufft -std=c++17 -arch=sm_75 \
-gencode arch=compute_61,code=sm_61 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_75,code=sm_75 #-DSPECTRALCN
PARAMSEW = -DC2=1.0 -DTAU=$(TAU) -DMONITOR=$(MONITOR) #-DDOUBLE  
PARAMSKPZ = -DC2=1.0 -DKPZ=1.0 -DTAU=$(TAU) -DMONITOR=$(MONITOR) -DNBINS=100 #-DDOUBLE  
PARAMSANH = -DC2=1.0 -DC4=1.0 -DTAU=$(TAU) -DMONITOR=$(MONITOR) #-DDOUBLE  
PARAMSPUREANH = -DC12=1.0 -DTAU=$(TAU) -DMONITOR=$(MONITOR) -DDt=$(Dt) #-DDOUBLE  

LDFLAGS = -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/math_libs/12.2/lib64 


activeinterface: ew.cu Makefile
	$(CXX) $(FLAGS) $(PARAMSPUREANH) ew.cu -o activeinterface $(LDFLAGS) $(INCLUDES) 

update_git:
	git add *.cu Makefile *.h *.sh README.md ; git commit -m "program update"; git push

clean:
	rm activeinterface
