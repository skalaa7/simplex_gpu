#define __CL_ENABLE_EXCEPTIONS
#include <omp.h>
#include "cl_util.hpp"
#include <CL/cl.hpp>
#include <fstream>
#include <iostream>
#define NUMOFVAR 1200
#define NUMOFSLACK 1200
#define ROWSIZE (NUMOFSLACK+1)
#define COLSIZE (NUMOFSLACK+NUMOFVAR+1)

bool checkOptimality(float wv[ROWSIZE*COLSIZE])
{
    for(int i=0;i<COLSIZE-1;i++)
    {
        if(wv[(ROWSIZE-1)*COLSIZE+i]<0)//min> max<
            return false;
    }
    return true;
}
bool isUnbounded(float wv[ROWSIZE*COLSIZE],int pivotCol)
{
    for(int j=0;j<ROWSIZE-1;j++)
    {
        if(wv[j*COLSIZE+pivotCol]>0)
            return false;
    }
    return true;
}
void makeMatrix(float wv[ROWSIZE*COLSIZE])
{
	for(int j=0;j<ROWSIZE; j++)
	{
		for(int i =0;i<COLSIZE;i++)
		{
			wv[j*COLSIZE+i]=0;
		}
	}
	fstream myFile;
    myFile.open("baza1202.txt",ios::in); //otvaram fajl u read modu
	if(myFile.is_open())
    {
        for(int j = 0; j < ROWSIZE; j++)
        {
            for(int i = 0; i< NUMOFVAR; i++)
            {
              myFile >> wv[j*COLSIZE+i];
            }
        }
		for(int j = 0;j< NUMOFSLACK;j++)
		{
			myFile >> wv[j*COLSIZE+COLSIZE-1];
		}
    }
    myFile.close();
    for(int j=0;j<ROWSIZE-1;j++)
    {
		
	wv[j*COLSIZE+NUMOFVAR+j]=1;
		
    }
}
int findPivotCol(float wv[ROWSIZE*COLSIZE])
{
     float minnegval=wv[(ROWSIZE-1)*COLSIZE+0];
       int loc=0;
        for(int i=1;i<COLSIZE-1;i++)
        {
            if(wv[(ROWSIZE-1)*COLSIZE+i]<minnegval)
            {
                minnegval=wv[(ROWSIZE-1)*COLSIZE+i];
                loc=i;
            }
        }
        return loc;
}
int findPivotRow(float wv[ROWSIZE*COLSIZE],int pivotCol)
{
    float rat[ROWSIZE-1];
    for(int j=0;j<ROWSIZE-1;j++)
        {
            if(wv[j*COLSIZE+pivotCol]>0)
            {
                rat[j]=wv[j*COLSIZE+COLSIZE-1]/wv[j*COLSIZE+pivotCol];
            }
            else
            {
                rat[j]=0;
            }
        }

        float minpozval=99999999;
        int loc=0;
        for(int j=0;j<ROWSIZE-1;j++)
        {
            if(rat[j]>0)
            {
                if(rat[j]<minpozval)
                {
                    minpozval=rat[j];
                    loc=j;
                }
            }
        }
        return loc;
}

void solutions(float wv[ROWSIZE*COLSIZE])
{
    for(int i=0;i<NUMOFVAR; i++)  //every basic column has the values, get it form B array
     {
        int count0 = 0;
        int index = 0;
        for(int j=0; j<ROWSIZE-1; j++)
        {
            if(wv[j*COLSIZE+i]==0.0)
            {
                count0 = count0+1;
            }
            else if(wv[j*COLSIZE+i]==1)
            {
                index = j;
            }


        }

        if(count0 == ROWSIZE - 2 )
        {
            cout<<"variable"<<i+1<<": "<<wv[index*COLSIZE+COLSIZE-1]<<endl;  //every basic column has the values, get it form B array
        }
        else
        {
            cout<<"variable"<<i+1<<": "<<0<<endl;
        }
    }

    cout<<""<<endl;
    cout<<endl<<"Optimal solution is "<<wv[(ROWSIZE-1)*COLSIZE+COLSIZE-1]<<endl;
}
void simplexCalculate(float wv[ROWSIZE*COLSIZE])
{

    //float minnegval;
    //float minpozval;
    //int loc;
    int pivotRow;
    int pivotCol;
    bool unbounded=false;
    float pivot;

    //float solVar[NUMOFVAR];

    while(!checkOptimality(wv))
    {
    	count++;
        pivotCol=findPivotCol(wv);

        if(isUnbounded(wv,pivotCol))
        {
            unbounded=true;
            break;
        }


        pivotRow=findPivotRow(wv,pivotCol);
	//cout<<count<<",pivot="<<wv[pivotRow][pivotCol]<<endl;
        pivot=wv[pivotRow][pivotCol];
	//s=omp_get_wtime();
    	
        doPivoting(wv,pivotRow,pivotCol,pivot);
       // s=omp_get_wtime()-s;
        //print(wv);

    }
    //Ispisivanje rezultata
    if(unbounded)
    {
        cout<<"Unbounded"<<endl;
    }
    else
    {
        //print(wv);

        solutions(wv);

    }
}
void init_gpu
{
	// get all platforms (drivers), e.g. NVIDIA
	std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);

    if (all_platforms.size()==0) {
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform=all_platforms[0];
    std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

    // get default device (CPUs, GPUs) of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }
	// use device[1] because that's a GPU; device[0] is the CPU
    cl::Device default_device=all_devices[1];
    std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";
	
	// a context is like a "runtime link" to the device and platform;
    // i.e. communication is possible
    cl::Context context({default_device});
	
	
	cl::Program program(context, cl_util::load_prog("pivot.cl"), true);
    if (program.build({default_device}) != CL_SUCCESS) {
        std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << std::endl;
        exit(1);
    }
	cl::make_kernel<int, int, int, cl::Buffer, cl::Buffer, cl::Buffer> pivot(prog, "pivot");
}
float wv[ROWSIZE][COLSIZE];
cl::Buffer buffer_newRow,buffer_pivotColVal, buffer_wv;
// create a queue (a queue of commands that the GPU will execute)
    cl::CommandQueue queue(context, default_device);
void doPivoting(float wv[ROWSIZE*COLSIZE],int pivotRow,int pivotCol,float pivot)
{
	float newRow[COLSIZE];
	float pivotColVal[ROWSIZE];
    for(int i=0;i<COLSIZE;i++)
        {
            newRow[i]=wv[pivotRow*COLSIZE+i]/pivot;
        }

        for(int j=0;j<ROWSIZE;j++)
        {
            pivotColVal[j]=wv[j*COLSIZE+pivotCol];
        }
	////gpu part
	queue.enqueueWriteBuffer(buffer_newRow,CL_TRUE,0,sizeof(float) * COLSIZE,newRow);
	queue.enqueueWriteBuffer(buffer_pivotColVal,CL_TRUE,0,sizeof(float) * ROWSIZE,pivotColVal);
	queue.enqueueWriteBuffer(buffer_wv,CL_TRUE,0,sizeof(float) * ROWSIZE*COLSIZE,wv);
	pivot(cl::EnqueueArgs(queue, cl::NDRange(ROWSIZE)),
	   	   pivotRow, ROWSIZE, COLSIZE, buffer_newRow, buffer_pivotColVal, buffer_wv);
	queue.finish();
	cl::copy(queue, buffer_wv, wv, wv+(ROWSIZE-1)*(COLSIZE-1));
        
}
int main() {
	init_gpu();
	makeMatrix(wv);
	
	
    // create buffers on device (allocate space on GPU)
    buffer_newRow = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float) * COLSIZE);
    buffer_pivotColVal = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float) * ROWSIZE);
    buffer_wv = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * ROWSIZE*COLSIZE);
	
	
	//main alghorithm
	simplexCalculate(wv);
	
	
	
	return 0;
}