.ORIG x3000
LD R6,stack_pointer     ; initialize the stack pointer
LD R1,entry
LD R2,start             ; x2000 写入地址 x0180 
STR R2,R1,#0            ; set up the keyboard interrupt vector table entry
LD R3,IE                ; 二进制0100 0000 0000 0000,使能位置为1,启用键盘中断
STI R3,KBSR             ; enable keyboard interrupts

begin                   ; start of actual user program to print ICS checkerboard
	AND R4,R4,#0           
	ADD R4,R4,#6
print_1                 ; 这里开始打印交错的ICS
	BRz end1
	LEA R0,ICS_1
	PUTS
	JSR DELAY       ; 延时，避免打印太快
       	ADD R4,R4,#-1   ; 计数器-1
	BRnzp print_1
end1	LD R0,enter
	OUT             ; 输出回车，准备打印下一行
	JSR DELAY

AND R4,R4,#0            ; 同理
ADD R4,R4,#5
print_2	
	BRz end2
	LEA R0,ICS_2
	PUTS
	JSR DELAY
	ADD R4,R4,#-1
	BRnzp print_2

end2	LD R0,enter
	OUT
	JSR DELAY
	BRnzp begin

HALT

DELAY   ST  R1, SaveR1    ;延时子程序
        LD  R1, COUNT
REP     ADD R1,R1,#-1
        BRp REP
        LD  R1, SaveR1
        RET

stack_pointer .fill x3000 ; 栈指针
entry    .fill x0180      ; 内存地址x0180（初始化为x2000）
start    .fill x2000      ; 初始值x2000（存入x0180）
IE       .fill x4000      ; 中断使能值（写入KBSR）
KBSR     .fill xFE00      ; 键盘状态寄存器地址
ICS_1    .STRINGZ "ICS   " 
ICS_2    .STRINGZ "   ICS" 
enter    .fill x000A      ; 换行符（ASCII码0x0A）
COUNT    .fill #25000      ; 延时循环次数
SaveR1   .BLKW 1          ; 保存R1的临时地址

.END
