#include <stdio.h>
#include "NUC100Series.h"

#define PLL_CLOCK       50000000


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
	
	GPIO_SetMode(PB, BIT2, GPIO_PMD_OUTPUT);
	PB2=1;

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

/*---------------------------------------------------------------------------------------------------------*/
/* Function: AdcSingleCycleScanModeTest                                                                    */
/*                                                                                                         */
/* Parameters:                                                                                             */
/*   None.                                                                                                 */
/*                                                                                                         */
/* Returns:                                                                                                */
/*   None.                                                                                                 */
/*                                                                                                         */
/* Description:                                                                                            */
/*   ADC single cycle scan mode test.                                                                      */
/*---------------------------------------------------------------------------------------------------------*/
void AdcSingleCycleScanModeTest()
{
    uint8_t  u8Option;
    uint32_t u32ChannelCount;
    int32_t  i32ConversionData;
		uint8_t  i, count=0, premode=0;

    printf("\n");
    printf("+----------------------------------------------------------------------+\n");
    printf("|                 ADC single cycle scan mode sample code               |\n");
    printf("+----------------------------------------------------------------------+\n");
		printf("\nPress 0 once to Reset\n");
		printf("Press 1 once to Collect 10 pieces of data for training\n");
		printf("Press 2 once to Collect 10 pieces of data for testing\n");
    while(1)
    {
        //printf("\n\nSelect input mode:\n");
        //printf("  [1] Single end input (channel 0, 1, 2 and 3)\n");
        //printf("  [2] Differential input (input channel pair 0 and 1)\n");
        //printf("  Other keys: exit single cycle scan mode test\n");
				
        u8Option = getchar();
				if(u8Option == '0'){
						if(premode != 0){
							premode = 1;
							count = 0;
							printf("\n}\n");
						}
						printf("\nPress 0 once to Reset\n");
						printf("Press 1 once to Collect 10 pieces of data for training\n");
						printf("Press 2 once to Collect 10 pieces of data for testing\n");
				}
        else if(u8Option == '1')
        {
						if(premode != 1){
							if(premode==2) printf("\n}\n");
							premode = 1;
							count = 0;
							printf("\nTraining Data:\n");
							printf("{");
						}
						
						if(count) printf(",");
						for(i=1; i<=10; i++)
						{
								
								if(i%10==1) printf("\n");
								printf("{");
								
								
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
										i32ConversionData = ADC_GET_CONVERSION_DATA(ADC, u32ChannelCount);
										if(u32ChannelCount!=2){
											printf("%d, ",i32ConversionData);
										} else {
											printf("%d}",i32ConversionData);
										}
								}
								
								if(i!=10) printf(",");
								CLK_SysTickDelay(1000000); //Delay 1 second.
						}
						count++;
        }else if(u8Option == '2')
				{
						if(premode != 2){
							if(premode==1) printf("\n}\n");
							premode = 2;
							count = 0;
							printf("\nTesting Data:\n");
							printf("{");
						}
						
						if(count) printf(",");
						for(i=1; i<=10; i++)
						{
								
								if(i%10==1) printf("\n");
								printf("{");
								
								
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
										i32ConversionData = ADC_GET_CONVERSION_DATA(ADC, u32ChannelCount);
										if(u32ChannelCount!=2){
											printf("%d, ",i32ConversionData);
										} else {
											printf("%d}",i32ConversionData);
										}
								}
								
								if(i!=10) printf(",");
								CLK_SysTickDelay(1000000); //Delay 1 second.
						}
						count++;
				}
					
    }
}

/*---------------------------------------------------------------------------------------------------------*/
/* MAIN function                                                                                           */
/*---------------------------------------------------------------------------------------------------------*/

int main(void)
{

    /* Unlock protected registers */
    SYS_UnlockReg();

    /* Init System, IP clock and multi-function I/O */
    SYS_Init();

    /* Lock protected registers */
    SYS_LockReg();

    /* Init UART0 for printf */
    UART0_Init();

    /*---------------------------------------------------------------------------------------------------------*/
    /* SAMPLE CODE                                                                                             */
    /*---------------------------------------------------------------------------------------------------------*/

    printf("\nSystem clock rate: %d Hz", SystemCoreClock);

    /* Single cycle scan mode test */
    AdcSingleCycleScanModeTest();

    /* Disable ADC module */
    ADC_Close(ADC);

    /* Disable ADC IP clock */
    CLK_DisableModuleClock(ADC_MODULE);

    /* Disable External Interrupt */
    NVIC_DisableIRQ(ADC_IRQn);

    printf("\nExit ADC sample code\n");

    while(1);

}
