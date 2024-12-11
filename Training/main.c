#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "NUC100Series.h"

#define PLL_CLOCK       50000000

/******************************************************************
 * dataset format setting
 ******************************************************************/

#define input_length 3 //Fixed Input length
#define train_data_num 210 //Total number of training data
#define test_data_num 70
#define target_num 7 //The number of output

#define train_data_num_each 30
#define test_data_num_each 10

/******************************************************************
 * Network Configuration - customized per network 
 ******************************************************************/
#define HiddenNodes 10

const float LearningRate = 0.11;
const float Momentum = 0.9;
const float InitialWeightMax = 0.5;
const float goal_acc = 0.98; //Target accuracy

// Create training dataset/output
// Your can put your training dataset here.
float train_data_input[train_data_num][input_length] = {
{128, 141, 51},{134, 147, 56},{148, 164, 64},{158, 180, 70},{160, 185, 71},{165, 192, 79},{163, 186, 74},{161, 195, 76},{159, 177, 77},{162, 192, 75},
{415, 426, 232},{422, 441, 235},{417, 419, 229},{423, 437, 234},{405, 414, 223},{418, 435, 228},{402, 415, 222},{411, 421, 225},{393, 405, 214},{400, 408, 217},
{403, 416, 216},{408, 416, 221},{402, 414, 220},{402, 402, 215},{402, 413, 219},{395, 401, 213},{401, 414, 216},{392, 399, 211},{399, 406, 217},{391, 399, 213},
{280, 464, 376},{280, 466, 375},{272, 445, 361},{244, 389, 304},{214, 336, 245},{204, 305, 218},{192, 271, 184},{187, 258, 166},{182, 245, 152},{183, 244, 148},
{182, 244, 153},{182, 259, 160},{187, 254, 164},{196, 275, 185},{197, 292, 199},{205, 309, 217},{223, 335, 250},{240, 377, 290},{273, 442, 350},{286, 462, 370},
{363, 580, 489},{351, 561, 462},{297, 487, 396},{270, 446, 366},{232, 388, 308},{221, 367, 285},{202, 339, 258},{192, 316, 229},{184, 299, 222},{179, 291, 206},
{846, 589, 385},{847, 588, 381},{806, 555, 364},{725, 503, 320},{647, 451, 280},{546, 385, 229},{490, 350, 204},{462, 335, 189},{437, 316, 178},{432, 316, 174},
{1025, 715, 490},{1011, 707, 481},{957, 664, 448},{864, 595, 395},{755, 518, 331},{660, 457, 282},{630, 444, 267},{589, 413, 247},{528, 379, 219},{478, 348, 193},
{384, 286, 153},{371, 286, 149},{366, 282, 144},{376, 288, 150},{388, 295, 152},{412, 316, 166},{448, 329, 182},{481, 351, 194},{531, 382, 216},{627, 445, 263},
{1110, 763, 462},{1070, 725, 436},{966, 648, 380},{939, 628, 362},{801, 537, 296},{727, 489, 262},{647, 432, 229},{577, 393, 202},{538, 368, 187},{514, 357, 180},
{336, 260, 119},{337, 262, 119},{341, 263, 116},{377, 284, 129},{402, 301, 137},{468, 338, 160},{522, 361, 179},{596, 410, 204},{644, 439, 228},{659, 446, 231},
{976, 659, 388},{901, 606, 348},{730, 490, 264},{609, 412, 210},{514, 350, 175},{476, 329, 160},{446, 315, 151},{416, 296, 140},{398, 280, 136},{370, 269, 124},
{356, 296, 128},{365, 304, 129},{405, 336, 144},{477, 388, 170},{577, 453, 212},{698, 546, 266},{838, 646, 336},{1026, 786, 434},{1099, 842, 476},{1138, 878, 501},
{1125, 867, 496},{1082, 829, 466},{997, 763, 419},{912, 694, 376},{817, 626, 325},{739, 566, 291},{641, 496, 246},{574, 443, 217},{534, 426, 199},{521, 411, 195},
{327, 279, 122},{321, 277, 118},{343, 292, 127},{380, 321, 133},{437, 355, 158},{488, 396, 178},{549, 436, 201},{654, 516, 249},{699, 547, 265},{755, 584, 297},
{1182, 1130, 681},{1170, 1108, 669},{1042, 991, 579},{929, 890, 495},{795, 768, 411},{729, 708, 367},{673, 656, 336},{624, 609, 306},{583, 578, 283},{553, 552, 264},
{399, 404, 181},{405, 409, 183},{418, 425, 190},{472, 473, 218},{521, 518, 244},{584, 583, 278},{652, 650, 318},{731, 716, 368},{783, 768, 401},{794, 780, 412},
{1079, 1033, 607},{1009, 963, 552},{899, 862, 478},{815, 789, 425},{736, 720, 374},{671, 654, 330},{611, 607, 297},{557, 553, 267},{530, 530, 254},{493, 498, 230},
{370, 568, 327},{352, 539, 310},{313, 488, 270},{279, 434, 236},{238, 369, 194},{197, 304, 150},{194, 288, 143},{190, 274, 135},{172, 242, 116},{163, 224, 104},
{336, 518, 289},{329, 492, 275},{282, 417, 228},{252, 364, 192},{237, 338, 174},{225, 308, 153},{217, 300, 142},{219, 292, 137},{210, 281, 132},{204, 261, 119},
{175, 211, 89},{181, 210, 93},{181, 220, 94},{190, 238, 104},{201, 257, 114},{206, 276, 126},{219, 313, 151},{226, 335, 164},{242, 355, 179},{264, 399, 204}
};
int train_data_output[train_data_num][target_num] = {0};

// Create testing dataset/output
// Your can put your testing dataset here.
float test_data_input[test_data_num][input_length] = {
{165, 192, 77},{160, 174, 73},{166, 194, 77},{154, 165, 71},{160, 187, 70},{151, 161, 67},{151, 165, 67},{143, 161, 61},{130, 142, 59},{122, 134, 58},
{263, 433, 360},{250, 412, 336},{201, 326, 256},{169, 257, 187},{154, 227, 150},{143, 204, 125},{140, 198, 113},{139, 178, 107},{134, 184, 100},{130, 171, 96},
{997, 709, 502},{1016, 725, 515},{916, 641, 450},{778, 544, 366},{608, 429, 271},{497, 364, 213},{406, 302, 173},{344, 273, 143},{307, 250, 130},{287, 234, 122},
{901, 614, 366},{906, 619, 371},{841, 570, 336},{765, 512, 299},{717, 479, 274},{627, 426, 240},{565, 398, 216},{535, 378, 202},{491, 361, 192},{455, 327, 172},
{1013, 779, 452},{981, 755, 431},{891, 685, 384},{811, 619, 338},{702, 542, 283},{613, 477, 242},{554, 440, 216},{492, 392, 190},{434, 357, 164},{410, 334, 156},
{1277, 1221, 795},{1243, 1190, 764},{1145, 1090, 681},{1064, 1013, 615},{924, 884, 509},{776, 740, 410},{654, 641, 334},{559, 553, 275},{495, 494, 241},{448, 449, 216},
{402, 614, 367},{375, 576, 338},{307, 466, 268},{254, 394, 218},{222, 336, 179},{202, 304, 156},{178, 262, 128},{168, 241, 111},{160, 220, 106},{163, 219, 103}
};
int test_data_output[test_data_num][target_num] = {0};

/******************************************************************
 * End Network Configuration
 ******************************************************************/


int ReportEvery10;
int RandomizedIndex[train_data_num];
long  TrainingCycle;
float Rando;
float Error;
float Accum;

float data_mean[3] ={0};
float data_std[3] ={0};

float Hidden[HiddenNodes];
float Output[target_num];
float HiddenWeights[input_length+1][HiddenNodes];
float OutputWeights[HiddenNodes+1][target_num];
float HiddenDelta[HiddenNodes];
float OutputDelta[target_num];
float ChangeHiddenWeights[input_length+1][HiddenNodes];
float ChangeOutputWeights[HiddenNodes+1][target_num];

int target_value;
int out_value;
int max;


/*---------------------------------------------------------------------------------------------------------*/
/* Define Function Prototypes                                                                              */
/*---------------------------------------------------------------------------------------------------------*/
void SYS_Init(void);
void UART0_Init(void);
void AdcSingleCycleScanModeTest(void);


void SYS_Init(void)
{
    /*---------------------------------------------------------------------------------------------------------*/
    /* Init System Clock                                                                                       */
    /*---------------------------------------------------------------------------------------------------------*/

    /* Enable Internal RC 22.1184MHz clock */
    CLK_EnableXtalRC(CLK_PWRCON_OSC22M_EN_Msk);

    /* Waiting for Internal RC clock ready */
    CLK_WaitClockReady(CLK_CLKSTATUS_OSC22M_STB_Msk);

    /* Switch HCLK clock source to Internal RC and HCLK source divide 1 */
    CLK_SetHCLK(CLK_CLKSEL0_HCLK_S_HIRC, CLK_CLKDIV_HCLK(1));

    /* Enable external XTAL 12MHz clock */
    CLK_EnableXtalRC(CLK_PWRCON_XTL12M_EN_Msk);

    /* Waiting for external XTAL clock ready */
    CLK_WaitClockReady(CLK_CLKSTATUS_XTL12M_STB_Msk);

    /* Set core clock as PLL_CLOCK from PLL */
    CLK_SetCoreClock(PLL_CLOCK);

    /* Enable UART module clock */
    CLK_EnableModuleClock(UART0_MODULE);

    /* Enable ADC module clock */
    CLK_EnableModuleClock(ADC_MODULE);

    /* Select UART module clock source */
    CLK_SetModuleClock(UART0_MODULE, CLK_CLKSEL1_UART_S_PLL, CLK_CLKDIV_UART(1));

    /* ADC clock source is 22.1184MHz, set divider to 7, ADC clock is 22.1184/7 MHz */
    CLK_SetModuleClock(ADC_MODULE, CLK_CLKSEL1_ADC_S_HIRC, CLK_CLKDIV_ADC(7));

    /*---------------------------------------------------------------------------------------------------------*/
    /* Init I/O Multi-function                                                                                 */
    /*---------------------------------------------------------------------------------------------------------*/

    /* Set GPB multi-function pins for UART0 RXD and TXD */
    SYS->GPB_MFP &= ~(SYS_GPB_MFP_PB0_Msk | SYS_GPB_MFP_PB1_Msk);
    SYS->GPB_MFP |= SYS_GPB_MFP_PB0_UART0_RXD | SYS_GPB_MFP_PB1_UART0_TXD;

    /* Disable the GPA0 - GPA3 digital input path to avoid the leakage current. */
    GPIO_DISABLE_DIGITAL_PATH(PA, 0xF);

    /* Configure the GPA0 - GPA3 ADC analog input pins */
    SYS->GPA_MFP &= ~(SYS_GPA_MFP_PA0_Msk | SYS_GPA_MFP_PA1_Msk | SYS_GPA_MFP_PA2_Msk | SYS_GPA_MFP_PA3_Msk) ;
    SYS->GPA_MFP |= SYS_GPA_MFP_PA0_ADC0 | SYS_GPA_MFP_PA1_ADC1 | SYS_GPA_MFP_PA2_ADC2 | SYS_GPA_MFP_PA3_ADC3 ;
    SYS->ALT_MFP1 = 0;

}

/*---------------------------------------------------------------------------------------------------------*/
/* Init UART                                                                                               */
/*---------------------------------------------------------------------------------------------------------*/
void UART0_Init()
{
    /* Reset IP */
    SYS_ResetModule(UART0_RST);

    /* Configure UART0 and set UART0 Baudrate */
    UART_Open(UART0, 115200);
}

void scale_data()
{
		float sum[3] = {0};
		int i, j;
		
		// Compute Data Mean
		for(i = 0; i < train_data_num; i++){
			for(j = 0; j < input_length; j++){
				sum[j] += train_data_input[i][j];
			}
		}
		for(j = 0; j < input_length ; j++){
			data_mean[j] = sum[j] / train_data_num;
			printf("MEAN: %.2f\n", data_mean[j]);
			sum[j] = 0.0;
		}
		
		// Compute Data STD
		for(i = 0; i < train_data_num; i++){
			for(j = 0; j < input_length ; j++){
				sum[j] += pow(train_data_input[i][j] - data_mean[j], 2);
			}
		}
		for(j = 0; j < input_length; j++){
			data_std[j] = sqrt(sum[j]/train_data_num);
			printf("STD: %.2f\n", data_std[j]);
			sum[j] = 0.0;
		}
}

void normalize(float *data)
{
    /*float sum = 0;
    int i;

    for(i=0; i<input_length; i++)
    {
        sum += data[i];
    }

    for(i=0; i<input_length; i++)
    {
        data[i] = data[i] / sum;
    }*/
	
		int i;
	
		for(i = 0; i < input_length; i++){
			data[i] = (data[i] - data_mean[i]) / data_std[i];
		}
}

int train_preprocess()
{
    int i;
    
    for(i = 0 ; i < train_data_num ; i++)
    {
        normalize(train_data_input[i]);
    }

    // Label of the training data
    for(i = 0 ; i < train_data_num ; i++)
    {
        if(i < train_data_num_each){
            train_data_output[i][0] = 1;
        }else if(i < train_data_num_each*2){
            train_data_output[i][1] = 1;
        }else if(i < train_data_num_each*3){
            train_data_output[i][2] = 1;
        }else if(i < train_data_num_each*4){
            train_data_output[i][3] = 1;
        }else if(i < train_data_num_each*5){
            train_data_output[i][4] = 1;
        }else if(i < train_data_num_each*6){
            train_data_output[i][5] = 1;
        }else if(i < train_data_num_each*7){
            train_data_output[i][6] = 1;
        }
    }
    return 0;
}

int test_preprocess()
{
    int i;

    for(i = 0 ; i < test_data_num ; i++)
    {
        normalize(test_data_input[i]);
    }

    // Label of the testing data
    for(i = 0 ; i < test_data_num ; i++)
    {
        if(i < test_data_num_each){
            test_data_output[i][0] = 1;
        }else if(i < test_data_num_each*2){
            test_data_output[i][1] = 1;
        }else if(i < test_data_num_each*3){
            test_data_output[i][2] = 1;
        }else if(i < test_data_num_each*4){
            test_data_output[i][3] = 1;
        }else if(i < test_data_num_each*5){
            test_data_output[i][4] = 1;
        }else if(i < test_data_num_each*6){
            test_data_output[i][5] = 1;
        }else if(i < test_data_num_each*7){
            test_data_output[i][6] = 1;
        }
    }
    return 0;
}

int data_setup()
{
    int i;
		//int j;
		int p, ret;
		unsigned int seed = 1;
	
		/* Set the ADC operation mode as single-cycle, input mode as single-end and
                 enable the analog input channel 0, 1 and 2 */
    ADC_Open(ADC, ADC_ADCR_DIFFEN_SINGLE_END, ADC_ADCR_ADMD_SINGLE_CYCLE, 0x7);

    /* Power on ADC module */
    ADC_POWER_ON(ADC);

    /* Clear the A/D interrupt flag for safe */
    ADC_CLR_INT_FLAG(ADC, ADC_ADF_INT);

    /* Start A/D conversion */
    ADC_START_CONV(ADC);

    /* Wait conversion done */
    while(!ADC_GET_INT_FLAG(ADC, ADC_ADF_INT));
		
		for(i = 0; i < 3; i++)
    {
				seed *= ADC_GET_CONVERSION_DATA(ADC, i);
    }
		seed *= 1000;
		printf("\nRandom seed: %d\n", seed);
    srand(seed);

    ReportEvery10 = 1;
    for( p = 0 ; p < train_data_num ; p++ ) 
    {    
        RandomizedIndex[p] = p ;
    }
		
		scale_data();
    ret = train_preprocess();
    ret |= test_preprocess();
    if(ret) //Error Check
        return 1;

    /*for(i=0;i<120;i++){
        printf("\ntrain DATA[%d] output: ",i+1);
        for (j = 0; j < target_num;j++)
            printf("%d ", train_data_output[i][j]);
    }*/

    /*for(i=0;i<40;i++){
        printf("\ntest DATA[%d] output: ",i+1);
        for (j = 0; j < target_num;j++)
            printf("%d ", test_data_output[i][j]);
    }*/
    return 0;
}

void run_train_data()
{
    int i, j, p;
    int correct=0;
    float accuracy = 0;
    printf("Train result:\n");
    for( p = 0 ; p < train_data_num ; p++ )
    { 
        max = 0;
        for (i = 1; i < target_num; i++) 
        {
            if (train_data_output[p][i] > train_data_output[p][max]) {
                max = i;
            }
        }
        target_value = max;
        
    /******************************************************************
    * Compute hidden layer activations
    ******************************************************************/

        for( i = 0 ; i < HiddenNodes ; i++ ) {    
            Accum = HiddenWeights[input_length][i] ;
            for( j = 0 ; j < input_length ; j++ ) {
                Accum += train_data_input[p][j] * HiddenWeights[j][i] ;
            }
            Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
        }

    /******************************************************************
    * Compute output layer activations and calculate errors
    ******************************************************************/

        for( i = 0 ; i < target_num ; i++ ) {    
            Accum = OutputWeights[HiddenNodes][i] ;
            for( j = 0 ; j < HiddenNodes ; j++ ) {
                Accum += Hidden[j] * OutputWeights[j][i] ;
            }
            Output[i] = 1.0/(1.0 + exp(-Accum)) ; 
        }
        
        max = 0;
        for (i = 1; i < target_num; i++) 
        {
            if (Output[i] > Output[max]) {
                max = i;
            }
        }
        out_value = max;

        if(out_value!=target_value)
            printf("Error --> Training Pattern: %d,Target : %d, Output : %d\n", p, target_value, out_value);
        else
            correct++;
        }
        // Calculate accuracy
        accuracy = (float)correct / train_data_num;
        printf ("Accuracy = %.2f /100 \n",accuracy*100);

}

void run_test_data()
{
    int i, j, p;
    int correct=0;
    float accuracy = 0;
    printf("Test result:\n");
    for( p = 0 ; p < test_data_num ; p++ )
    { 
        max = 0;
        for (i = 1; i < target_num; i++) 
        {
            if (test_data_output[p][i] > test_data_output[p][max]) {
                max = i;
            }
        }
        target_value = max;
        
    /******************************************************************
    * Compute hidden layer activations
    ******************************************************************/

        for( i = 0 ; i < HiddenNodes ; i++ ) {    
            Accum = HiddenWeights[input_length][i] ;
            for( j = 0 ; j < input_length ; j++ ) {
                Accum += test_data_input[p][j] * HiddenWeights[j][i] ;
            }
            Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
        }

    /******************************************************************
    * Compute output layer activations and calculate errors
    ******************************************************************/

        for( i = 0 ; i < target_num ; i++ ) {    
            Accum = OutputWeights[HiddenNodes][i] ;
            for( j = 0 ; j < HiddenNodes ; j++ ) {
                Accum += Hidden[j] * OutputWeights[j][i] ;
            }
            Output[i] = 1.0/(1.0 + exp(-Accum)) ; 
        }
        max = 0;
        for (i = 1; i < target_num; i++) 
        {
            if (Output[i] > Output[max]) {
                max = i;
            }
        }
        out_value = max;

        if(out_value!=target_value)
            printf("Error --> Training Pattern: %d,Target : %d, Output : %d\n", p, target_value, out_value);
        else
            correct++;
        }
        // Calculate accuracy
        accuracy = (float)correct / test_data_num;
        printf ("Accuracy = %.2f /100 \n",accuracy*100);
}

float Get_Train_Accuracy()
{
    int i, j, p;
    int correct = 0;
		float accuracy = 0;
    for (p = 0; p < train_data_num; p++)
    {
/******************************************************************
* Compute hidden layer activations
******************************************************************/

        for( i = 0 ; i < HiddenNodes ; i++ ) {    
            Accum = HiddenWeights[input_length][i] ;
            for( j = 0 ; j < input_length ; j++ ) {
                Accum += train_data_input[p][j] * HiddenWeights[j][i] ;
            }
            Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
        }

/******************************************************************
* Compute output layer activations
******************************************************************/

        for( i = 0 ; i < target_num ; i++ ) {    
            Accum = OutputWeights[HiddenNodes][i] ;
            for( j = 0 ; j < HiddenNodes ; j++ ) {
                Accum += Hidden[j] * OutputWeights[j][i] ;
            }
            Output[i] = 1.0/(1.0 + exp(-Accum)) ; 
        }
        //get target value
        max = 0;
        for (i = 1; i < target_num; i++) 
        {
            if (train_data_output[p][i] > train_data_output[p][max]) {
                max = i;
            }
        }
        target_value = max;
        //get output value
        max = 0;
        for (i = 1; i < target_num; i++) 
        {
            if (Output[i] > Output[max]) {
                max = i;
            }
        }
        out_value = max;
        //compare output and target
        if (out_value==target_value)
        {
            correct++;
        }
    }

    // Calculate accuracy
    accuracy = (float)correct / train_data_num;
    return accuracy;
}

void load_weight()
{
    int i,j;
    printf("\n=======Hidden Weight=======\n");
    printf("{");
    for(i = 0; i <= input_length ; i++)
    {
        printf("{");
        for (j = 0; j < HiddenNodes; j++)
        {
            if(j!=HiddenNodes-1){
                printf("%f,", HiddenWeights[i][j]);
            }else{
                printf("%f", HiddenWeights[i][j]);
            }
        }
        if(i!=input_length){
            printf("},\n");
        }else {
            printf("}");
        }
    }
    printf("}\n");

    printf("\n=======Output Weight=======\n");

    for(i = 0; i <= HiddenNodes ; i++)
    {
        printf("{");
        for (j = 0; j < target_num; j++)
        {
            if(j!=target_num-1){
                printf("%f,", OutputWeights[i][j]);
            }else{
                printf("%f", OutputWeights[i][j]);
            }
        }
        if(i!=HiddenNodes){
            printf("},\n");
        }else {
            printf("}");
        }
    }
    printf("}\n");
}

void AdcSingleCycleScanModeTest()
{
		int i, j;
    uint32_t u32ChannelCount;
    float single_data_input[3];
		char output_string[10] = {NULL};

    printf("\n");	
		printf("[Phase 3] Start Prediction ...\n\n");
		PB2=1;
    while(1)
    {
			
				/* Set the ADC operation mode as single-cycle, input mode as single-end and
                 enable the analog input channel 0, 1, 2 and 3 */
        ADC_Open(ADC, ADC_ADCR_DIFFEN_SINGLE_END, ADC_ADCR_ADMD_SINGLE_CYCLE, 0x7);

        /* Power on ADC module */
        ADC_POWER_ON(ADC);

        /* Clear the A/D interrupt flag for safe */
        ADC_CLR_INT_FLAG(ADC, ADC_ADF_INT);

        /* Start A/D conversion */
        ADC_START_CONV(ADC);

        /* Wait conversion done */
        while(!ADC_GET_INT_FLAG(ADC, ADC_ADF_INT));

        for(u32ChannelCount = 0; u32ChannelCount < 3; u32ChannelCount++)
        {
            single_data_input[u32ChannelCount] = ADC_GET_CONVERSION_DATA(ADC, u32ChannelCount);
        }
				normalize(single_data_input);
						

				// Compute hidden layer activations
				for( i = 0 ; i < HiddenNodes ; i++ ) {    
						Accum = HiddenWeights[input_length][i] ;
						for( j = 0 ; j < input_length ; j++ ) {
								Accum += single_data_input[j] * HiddenWeights[j][i] ;
						}
						Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
				}

				// Compute output layer activations
				for( i = 0 ; i < target_num ; i++ ) {    
						Accum = OutputWeights[HiddenNodes][i] ;
						for( j = 0 ; j < HiddenNodes ; j++ ) {
								Accum += Hidden[j] * OutputWeights[j][i] ;
						}
						Output[i] = 1.0/(1.0 + exp(-Accum)) ; 
				}
						
				max = 0;
				for (i = 1; i < target_num; i++) 
				{
						if (Output[i] > Output[max]) {
								max = i;
						}
				}
				out_value = max;
				
				switch(out_value){
						case 0:
								strcpy(output_string, "Ambient");
								break;
						case 1:
								strcpy(output_string, "Blue");	
								break;
						case 2:
								strcpy(output_string, "Magenta");	
								break;
						case 3:
								strcpy(output_string, "Red");
								break;
						case 4:
								strcpy(output_string, "Orange");	
								break;
						case 5:
								strcpy(output_string, "Yellow");	
								break;
						case 6:
								strcpy(output_string, "Green");
								break;
				}
				
				printf("\rPrediction output: Output: %-8s", output_string);
				CLK_SysTickDelay(500000);
				/*for(i=0;i<target_num;i++)
				{
						printf("%f ",Output[i]);
				}
				printf("\n%d\n", out_value);*/

    }
}

/*---------------------------------------------------------------------------------------------------------*/
/* MAIN function                                                                                           */
/*---------------------------------------------------------------------------------------------------------*/

int main(void)
{
		int i, j, p, q, r;
    float accuracy=0;

    /* Unlock protected registers */
    SYS_UnlockReg();

    /* Init System, IP clock and multi-function I/O */
    SYS_Init();

    /* Lock protected registers */
    SYS_LockReg();

    /* Init UART0 for printf */
    UART0_Init();
	
	  GPIO_SetMode(PB, BIT2, GPIO_PMD_OUTPUT);
	  PB2=0;
	
		printf("\n+-----------------------------------------------------------------------+\n");
    printf("|                        LAB8 - Machine Learning                        |\n");
    printf("+-----------------------------------------------------------------------+\n");
		printf("System clock rate: %d Hz\n", SystemCoreClock);

    printf("\n[Phase 1] Initialize DataSet ...");
	  /* Data Init (Input / Output Preprocess) */
		if(data_setup()){
        printf("[Error] Datasets Setup Error\n");
        return 0;
    }else
				printf("Done!\n\n");
		
		printf("[Phase 2] Start Model Training ...\n");
		// Initialize HiddenWeights and ChangeHiddenWeights 
    for( i = 0 ; i < HiddenNodes ; i++ ) {    
        for( j = 0 ; j <= input_length ; j++ ) { 
            ChangeHiddenWeights[j][i] = 0.0 ;
            Rando = (float)((rand() % 100))/100;
            HiddenWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
        }
    }

    // Initialize OutputWeights and ChangeOutputWeights
    for( i = 0 ; i < target_num ; i ++ ) {    
        for( j = 0 ; j <= HiddenNodes ; j++ ) {
            ChangeOutputWeights[j][i] = 0.0 ;  
            Rando = (float)((rand() % 100))/100;        
            OutputWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
        }
    }

    // Begin training 
    for(TrainingCycle = 1 ; TrainingCycle < 2147483647 ; TrainingCycle++)
    {
        Error = 0.0 ;

        // Randomize order of training patterns
        for( p = 0 ; p < train_data_num ; p++) {
            q = rand()%train_data_num;
            r = RandomizedIndex[p] ; 
            RandomizedIndex[p] = RandomizedIndex[q] ; 
            RandomizedIndex[q] = r ;
        }

        // Cycle through each training pattern in the randomized order
        for( q = 0 ; q < train_data_num ; q++ ) 
        {    
            p = RandomizedIndex[q];

            // Compute hidden layer activations
            for( i = 0 ; i < HiddenNodes ; i++ ) {    
                Accum = HiddenWeights[input_length][i] ;
                for( j = 0 ; j < input_length ; j++ ) {
                    Accum += train_data_input[p][j] * HiddenWeights[j][i] ;
                }
                Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
            }

            // Compute output layer activations and calculate errors
            for( i = 0 ; i < target_num ; i++ ) {    
                Accum = OutputWeights[HiddenNodes][i] ;
                for( j = 0 ; j < HiddenNodes ; j++ ) {
                    Accum += Hidden[j] * OutputWeights[j][i] ;
                }
                Output[i] = 1.0/(1.0 + exp(-Accum)) ;   
                OutputDelta[i] = (train_data_output[p][i] - Output[i]) * Output[i] * (1.0 - Output[i]) ;   
                Error += 0.5 * (train_data_output[p][i] - Output[i]) * (train_data_output[p][i] - Output[i]) ;
            }

            // Backpropagate errors to hidden layer
            for( i = 0 ; i < HiddenNodes ; i++ ) {    
                Accum = 0.0 ;
                for( j = 0 ; j < target_num ; j++ ) {
                    Accum += OutputWeights[i][j] * OutputDelta[j] ;
                }
                HiddenDelta[i] = Accum * Hidden[i] * (1.0 - Hidden[i]) ;
            }

            // Update Input-->Hidden Weights
            for( i = 0 ; i < HiddenNodes ; i++ ) {     
                ChangeHiddenWeights[input_length][i] = LearningRate * HiddenDelta[i] + Momentum * ChangeHiddenWeights[input_length][i] ;
                HiddenWeights[input_length][i] += ChangeHiddenWeights[input_length][i] ;
                for( j = 0 ; j < input_length ; j++ ) { 
                    ChangeHiddenWeights[j][i] = LearningRate * train_data_input[p][j] * HiddenDelta[i] + Momentum * ChangeHiddenWeights[j][i];
                    HiddenWeights[j][i] += ChangeHiddenWeights[j][i] ;
                }
            }

            // Update Hidden-->Output Weights
            for( i = 0 ; i < target_num ; i ++ ) {    
                ChangeOutputWeights[HiddenNodes][i] = LearningRate * OutputDelta[i] + Momentum * ChangeOutputWeights[HiddenNodes][i] ;
                OutputWeights[HiddenNodes][i] += ChangeOutputWeights[HiddenNodes][i] ;
                for( j = 0 ; j < HiddenNodes ; j++ ) {
                    ChangeOutputWeights[j][i] = LearningRate * Hidden[j] * OutputDelta[i] + Momentum * ChangeOutputWeights[j][i] ;
                    OutputWeights[j][i] += ChangeOutputWeights[j][i] ;
                }
            }
        }
        accuracy = Get_Train_Accuracy();

        // Every 10 cycles send data to terminal for display
        ReportEvery10 = ReportEvery10 - 1;
        if (ReportEvery10 == 0)
        {
            
            printf ("\nTrainingCycle: %ld\n",TrainingCycle);
            printf ("Error = %.5f\n",Error);
            printf ("Accuracy = %.2f /100 \n",accuracy*100);
            //run_train_data();

            if (TrainingCycle==1)
            {
                ReportEvery10 = 9;
            }
            else
            {
                ReportEvery10 = 10;
            }
        }

        // If error rate is less than pre-determined threshold then end
        if( accuracy >= goal_acc ) break ;
    }

    printf ("\nTrainingCycle: %ld\n",TrainingCycle);
    printf ("Error = %.5f\n",Error);
    run_train_data();
    printf ("Training Set Solved!\n");
    printf ("--------\n"); 
    printf ("Testing Start!\n ");
    run_test_data();
    printf ("--------\n"); 
    ReportEvery10 = 1;
    load_weight();
		
		printf("\nModel Training Phase has ended.\n");

    /* Single cycle scan mode test */
    AdcSingleCycleScanModeTest();

    while(1);
}
