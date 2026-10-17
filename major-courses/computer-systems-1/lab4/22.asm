  			.ORIG 		X3000			;
;初始化空白棋盘init
			LD		R3,DATA			;指向开辟数据区的首地址
			LD		R2,SIX			;R2作为初始化六行的计数器
			ADD 		R2,R2,#1		;
LOOPOUT0		LD		R1,SIX			;R1作为初始化六列的计数器
			ADD 		R2,R2,#-1		;
			BRp		LOOPIN0			;R2>0，继续循环
			BRz 		NEXT0			;循环结束，来到下一部分	
LOOPIN0			AND 		R4,R4,#0		;
			
			STR		R4,R3,#0		;棋盘初始化为0
			ADD		R3,R3,#1		;
			ADD		R1,R1,#-1		;
			BRp 		LOOPIN0			;
			BRz		LOOPOUT0		;
 
 
 
NEXT0			JSR		PRINT			;打印空白棋盘	
 
			
Input			JSR		IsFull 			;判断棋盘是否满了（和局）
			ADD		R0,R0,#0		;
			BRp		END			;
			
			JSR		Iscorrect1		;提示用户1输入并判断输入是否正确，返回用户输入的棋子坐标
			JSR		PRINT			;打印用户1下棋后的棋盘
 
			JSR		IsSHU1			;
			ADD		R6,R6,#0		;
			BRp		END			;
 
			JSR		IsLine1			;
			ADD		R6,R6,#0		;
			BRp		END			;
 
 
			AND		R6,R6,#0		;
			ADD		R6,R6,#1		;
			JSR		IsXie1			;
			ADD		R6,R6,#0		;
			BRp		END			;
 
			AND		R6,R6,#0		;
			ADD		R6,R6,#1		;
			JSR		IsXie2			;
			ADD		R6,R6,#0		;
			BRp		END			;
 
			
			AND		R6,R6,#0		;
			ADD		R6,R6,#1		;
			JSR		IsXie3			;
			ADD		R6,R6,#0		;
			BRp		END			;
 
			AND		R6,R6,#0		;
			ADD		R6,R6,#1		;
			JSR		IsXie4			;
			ADD		R6,R6,#0		;
			BRp		END			;
 
			LD		R0,BLANKLINE		;
			TRAP		X21			;
		
			JSR		Iscorrect2		;提示用户2输入并判断输入是否正确，返回用户输入的棋子坐标
			JSR		PRINT			;
 
			JSR		IsSHU2			;
			ADD		R6,R6,#0		;
			BRp		END			;
 
			JSR		IsLine2			;
			ADD		R6,R6,#0		;
			BRp		END			;
 
			AND		R6,R6,#0		;
			ADD		R6,R6,#-1		;
			JSR		IsXie1			;
			ADD		R6,R6,#0		;
			BRp		END			;
 
			AND		R6,R6,#0		;
			ADD		R6,R6,#-1		;
			JSR		IsXie2			;
			ADD		R6,R6,#0		;
			BRp		END			;
 
			
			AND		R6,R6,#0		;
			ADD		R6,R6,#-1		;
			JSR		IsXie3			;
			ADD		R6,R6,#0		;
			BRp		END			;
 
			AND		R6,R6,#0		;
			ADD		R6,R6,#-1		;
			JSR		IsXie4			;
			ADD		R6,R6,#0		;
			BRp		END			;
 
			LD		R0,BLANKLINE		;
			TRAP		X21			;
 
			BRnzp		Input			;
 
 
END			HALT
 
 
SIX			.fill 		x0006			;		
BLANK			.fill		x0020			;
BLANKLINE 		.fill		x000A			;
LINE 			.fill  		x002D			;	
user1			.stringz   "Player 1, choose a column:" ;
user2			.stringz   "Player 2, choose a column:" ;
Wrong			.stringz   "Invalid move. Try again."	;
Ascii0			.fill		x0030			;
Ascii6			.fill		x0036			;
DATA			.FILL 		x3500			;开辟一个50大小的空间来存储棋盘数据
DATAEND			.fill		x351E			;
CHARO			.fill    	x004F			;
CHARX			.fill		x0058			;
SAVER0			.BLKW		1			;
SAVER1			.BLKW		1			;
SAVER2			.BLKW		1			;
SAVER3			.BLKW		1			;
SAVER4			.BLKW		1			;
SAVER5			.BLKW		1			;
SAVER6			.BLKW		1			;
SAVER7			.BLKW		1			;
ASCII			.fill		-48			;
 
 
 
;打印棋盘
PRINT			ST 		R0,SAVER0		;
			ST		R1,SAVER1		;
			ST		R2,SAVER2		;
			ST		R3,SAVER3		;
			ST		R4,SAVER4		;
			ST		R7,SAVER7		;
			LD 		R3,DATA			;R3装载数据区首地址
			LD		R2,SIX			;R2作为输出六行的计数器
			ADD 		R2,R2,#1		;
LOOPOUT			LD		R1,SIX			;R1作为输出六列的计数器
			LD  		R0,BLANKLINE		;
			TRAP 		X21			;输出换行符
			ADD 		R2,R2,#-1		;
			BRp		LOOPIN			;R2>0，继续循环
			BRz 		FINISH			;循环结束，来到下一部分	
			
LOOPIN			LDR		R4,R3,#0		;R4装在数据区首地址的值
			BRz		LOADLINE		;如果R4=0，输出"-"
			BRp	 	LOADO			;如果R4>0，输出"O"
			BRn 		LOADX			;如果R4<0，输出"X"
			
LOADLINE		LD		R0,LINE			;
			TRAP		x21			;输出"-"
			ADD 		R3,R3,#1		;
			BRnzp		NEXTPART		;		
 
LOADO			LD		R0,CHARO		;
			TRAP 		X21			;输出"O"
			ADD		R3,R3,#1		;
			BRnzp		NEXTPART		;
 
LOADX			LD		R0,CHARX		;
			TRAP 		X21			;输出"X"
			ADD 		R3,R3,#1		;
			BRnzp		NEXTPART		;
 
NEXTPART		LD	 	R0,BLANK		;
			TRAP   		X21			;输出空格	
			ADD		R1,R1,#-1		;
			BRp 		LOOPIN			;
			BRz		LOOPOUT			;
FINISH			LD 		R0,SAVER0		;
			LD		R1,SAVER1		;
			LD		R2,SAVER2		;
			LD		R3,SAVER3		;
			LD		R4,SAVER4		;
			LD		R7,SAVER7		;
			RET
 
 
 
;判断用户1输入是否合理，即用户落子的列在1-6之间	
Iscorrect1		ST		R1,SAVER1		;
			ST		R2,SAVER2		;
			ST		R7,SAVER7		;
			ST		R3,SAVER3		;
			ST		R5,SAVER5		;
			BRnzp		NEXTPART1		;
 
WrongInput1		LD		R0,BLANKLINE		;
			TRAP		X21			;
			LEA      	R0,Wrong		;
			PUTS					;
			LD		R0,BLANKLINE		;
			TRAP 		X21			;
			
NEXTPART1		LEA		R0,user1		;
			PUTS					;提示用户1落子
			TRAP		X23			;用户输入落子的列
		
			LD 		R1,Ascii0		;
			NOT 		R1,R1			;
			ADD		R1,R1,#1		;
			ADD 		R2,R0,R1		;	
			BRnz		WrongInput1		;用户输入<1，重新输入	
			LD		R2,Ascii6		;
			NOT 		R2,R2			;
			ADD		R2,R2,#1		;
			ADD	 	R1,R0,R2		;
			BRp		WrongInput1		;用户输入>6,重新输入
			TRAP		X21			;回显用户输入
			LD		R6,ASCII		;
			ADD		R0,R0,R6		;进行用户的输入Ascii的转换
			LD		R3,DATAEND		;
			ADD		R3,R3,R0		;
			ADD 		R3,R3,#-1		;R3指向用户输入的列的第6行
 
IsNull1			LDR		R4,R3,#0		;判断该点处的状态
			BRz		Change1			;
			BRnp		LastLine1		;
		
Change1			ADD		R0,R3,#0		;
			AND		R4,R4,#0		;
			ADD		R4,R4,#1		;
			STR		R4,R3,#0		;
			BRnzp		Done1			;	
 
LastLine1		ADD		R3,R3,#-6		;
			LDR		R4,R3,#0		;
			BRnp		LastLine1		;
			LD		R5,DATA			;
			NOT		R5,R5			;
			ADD		R5,R5,#1		;
			ADD		R5,R3,R5		;
			BRn		WrongInput1		;
			ADD		R0,R3,#0		;
			AND		R4,R4,#0		;
			ADD		R4,R4,#1		;
			STR		R4,R3,#0		;
			
			
Done1			LD		R1,SAVER1		;
			LD		R2,SAVER2		;
			LD		R7,SAVER7		;
			LD		R3,SAVER3		;
			LD		R4,SAVER4		;
			LD		R5,SAVER5		;
			RET
 
 
 
 
;判断棋盘是否已经满了（和局）
IsFull			ST		R1,SAVER1		;
			ST		R2,SAVER2		;
			ST		R3,SAVER3		;
			ST 		R4,SAVER4		;
			ST		R7,SAVER7		;
			LD		R3,DATA			;指向开辟数据区的首地址
			LD		R2,SIX			;R2作为六行的计数器
			ADD 		R2,R2,#1		;
LOOPOUT1		LD		R1,SIX			;R1作为六列的计数器
			ADD 		R2,R2,#-1		;
			BRp		LOOPIN1			;R2>0，继续循环
			BRz 		Full			;循环结束，来到下一部分	
LOOPIN1			LDR		R4,R3,#0		;
			BRz		NotFull			;
			ADD		R3,R3,#1		;
			ADD		R1,R1,#-1		;
			BRp 		LOOPIN1			;
			BRz		LOOPOUT1		;
 
Full			AND		R0,R0,#0		;
			ADD		R0,R0,#1		;					
			BRnzp		DONE			;
 
NotFull			AND		R0,R0,#0		;
			BRnzp		DONE			;
 
DONE			LD		R1,SAVER1		;
			LD		R2,SAVER2		;
			LD 		R3,SAVER3		;
			LD		R4,SAVER4		;
			LD		R7,SAVER7		;
			RET
 
;判断用户2输入是否合理，即用户落子的列在1-6之间	
Iscorrect2		ST		R1,SAVER1		;
			ST		R2,SAVER2		;
			ST		R7,SAVER7		;
			ST		R3,SAVER3		;
			ST		R5,SAVER5		;
			BRnzp		NEXTPART2		;
 
WrongInput2		LD		R0,BLANKLINE		;
			TRAP		X21			;
			LEA      	R0,Wrong		;
			PUTS					;
			LD		R0,BLANKLINE		;
			TRAP 		X21			;
			
NEXTPART2		LEA		R0,user2		;
			PUTS					;提示用户2落子
			TRAP		X23			;用户输入落子的列
	
			LD 		R1,Ascii0		;
			NOT 		R1,R1			;
			ADD		R1,R1,#1		;
			ADD 		R2,R0,R1		;	
			BRnz		WrongInput2		;用户输入<1，重新输入	
			LD		R2,Ascii6		;
			NOT 		R2,R2			;
			ADD		R2,R2,#1		;
			ADD	 	R1,R0,R2		;
			BRp		WrongInput2		;用户输入>6,重新输入
			TRAP		X21			;回显用户输入
 
			LD		R6,ASCII		;
			ADD		R1,R0,R6		;进行用户的输入Ascii的转换
			LD		R3,DATAEND		;
			ADD		R3,R3,R1		;
			ADD 		R3,R3,#-1		;R3指向用户输入的列的第6行
 
IsNull2			LDR		R4,R3,#0		;判断该点处的状态
			BRz		ChangeN1		;
			BRnp		LastLine2		;
		
ChangeN1		ADD		R0,R3,#0		;
			AND		R4,R4,#0		;
			ADD		R4,R4,#-1		;
			STR		R4,R3,#0		;
			BRnzp		Done2			;	
 
LastLine2		ADD		R3,R3,#-6		;	
			LDR		R4,R3,#0		;
			BRnp		LastLine2		;
			LD		R5,DATA			;
			NOT		R5,R5			;
			ADD		R5,R5,#1		;
			ADD		R5,R3,R5		;
			BRn		WrongInput2		;
			ADD		R0,R3,#0		;
			AND		R4,R4,#0		;
			ADD		R4,R4,#-1		;
			STR		R4,R3,#0		;
			
Done2			LD		R1,SAVER1		;
			LD		R2,SAVER2		;
			LD		R7,SAVER7		;
			LD		R3,SAVER3		;
			LD  		R5,SAVER5		;
			RET
 
 
;判断用户1是否赢了比赛（竖）
IsSHU1			ST		R1,SR1			;
			ST		R2,SR2			;
			ST		R3,SR3			;
			ST 		R4,SR4			;
			ST		R7,SR7			;
			ST		R0,SR0			;
			ST		R5,SR5			;
 
			AND		R4,R4,#0		;
			AND		R6,R6,#0		;
			LD		R3,SR0			;
			LD		R1,NFOUR		;
SHU1			LDR		R2,R3,#0		;
			BRp		A1			;
			BRnz		FINISH1			;
 
A1			ADD		R4,R4,#1		;
			ADD		R2,R4,R1		;
			BRz		WIN1			;
			ADD		R3,R3,#6		;
			LD		R5,DATAend		;
			NOT		R5,R5			;
			ADD		R5,R5,#1		;
			ADD		R5,R3,R5		;
			BRnz		SHU1			;
			BRp		FINISH1			;			
 
			
WIN1			LEA		R0,U1WIN		;
			PUTS					;
			ADD		R6,R6,#1		;
			BRnzp		FINISH1			;
		
		
FINISH1			LD		R1,SR1			;
			LD		R2,SR2			;
			LD		R3,SR3			;
			LD 		R4,SR4			;
			LD		R7,SR7			;
			LD		R0,SR0			;
			LD		R5,SR5			;
 
			RET	
 
 
;判断用户2是否赢了比赛（竖）
IsSHU2			ST		R1,SR1			;
			ST		R2,SR2			;
			ST		R3,SR3			;
			ST 		R4,SR4			;
			ST		R7,SR7			;
			ST		R0,SR0			;
			ST		R5,SR5			;
 
			AND		R4,R4,#0		;
			AND		R6,R6,#0		;
			LD		R3,SR0			;
			LD		R1,NFOUR		;
SHU2			LDR		R2,R3,#0		;
			BRn		A2			;
			BRzp		FINISH2			;
 
A2			ADD		R4,R4,#1		;
			ADD		R2,R4,R1		;
			BRz		WIN2			;
			ADD		R3,R3,#6		;
			LD		R5,DATAend		;
			NOT		R5,R5			;
			ADD		R5,R5,#1		;
			ADD		R5,R3,R5		;
			BRnz		SHU2			;
			BRp		FINISH2			;			
 
 
WIN2			LEA		R0,U2WIN		;
			PUTS					;
			ADD		R6,R6,#1		;
			BRnzp		FINISH2			;
	
		
FINISH2			LD		R1,SR1			;
			LD		R2,SR2			;
			LD		R3,SR3			;
			LD 		R4,SR4			;
			LD		R7,SR7			;
			LD		R0,SR0			;
			LD		R5,SR5			;
			RET	
 
 
FOUR			.fill		x0004			;
NFOUR			.fill		#-4			;	
U1WIN			.stringz	"Player 1 Wins."	;
U2WIN			.stringz	"Player 2 Wins."	;
SR0			.BLKW		1			;
SR1			.BLKW		1			;
SR2			.BLKW		1			;
SR3			.BLKW		1			;
SR4			.BLKW		1			;
SR5			.BLKW		1			;
SR6			.BLKW		1			;
SR7			.BLKW		1			;
Data			.fill		x351E			;
DATAend			.fill		x3523			;
 
 
;判断是否用户1是否赢了比赛（横）
IsLine1			ST		R1,SR1			;
			ST		R2,SR2			;
			ST		R3,SR3			;
			ST 		R4,SR4			;
			ST		R7,SR7			;
			ST		R0,SR0			;
			ST		R5,SR5			;
 
		
			AND		R5,R5,#0		;
			AND		R6,R6,#0		;
 
			LEA		R1,RIGHT		;
	
R			LDR		R4,R1,#0		;
			NOT		R2,R4			;
			ADD		R2,R2,#1		;
			ADD		R2,R2,R0		;
			BRz		right			;
			ADD		R1,R1,#1		;
			ADD		R5,R5,#1		;
			LD		R2,Eighteen		;
			ADD   		R2,R5,R2		;
			BRz		left			;
			BRnp		R			;
 
 
right			AND		R5,R5,#0		;
			LD		R4,FOUR			;
			LD		R3,SR0			;
			LD		R1,NFOUR		;
 
HENG1			LDR		R2,R3,#0		;
			BRp		Add1			;
			BRnz		Finish1			;
 
 
Add1			ADD		R5,R5,#1		;
			ADD		R2,R5,R1		;
			BRz		WIN1			;
			ADD		R4,R4,#-1		;
			BRz		Finish1			;
			ADD		R3,R3,#1		;
			BRp		HENG1			;
 
 
 
left			AND		R5,R5,#0		;
			LD		R4,FOUR			;
			LD		R3,SR0			;
			LD		R1,NFOUR		;
 
HENG2			LDR		R2,R3,#0		;
			BRp		Add2			;
			BRnz		Finish1			;
 
 
Add2			ADD		R5,R5,#1		;
			ADD		R2,R5,R1		;
			BRz		WIN1			;
			ADD		R4,R4,#-1		;
			BRz		Finish1			;
			ADD		R3,R3,#-1		;
			BRp		HENG2			;
 
 
Win1			LEA		R0,U1WIN		;
			PUTS					;
			ADD		R6,R6,#1		;
			BRnzp		Finish1			;
	
 
 
Finish1			LD		R1,SR1			;
			LD		R2,SR2			;
			LD		R3,SR3			;
			LD 		R4,SR4			;
			LD		R7,SR7			;
			LD		R0,SR0			;
			LD		R5,SR5			;
			RET
 
 
;判断是否用户2是否赢了比赛（横）
IsLine2			ST		R1,SR1			;
			ST		R2,SR2			;
			ST		R3,SR3			;
			ST 		R4,SR4			;
			ST		R7,SR7			;
			ST		R0,SR0			;
			ST		R5,SR5			;
 
		
			AND		R5,R5,#0		;
			AND		R6,R6,#0		;
 
			LEA		R1,RIGHT		;
	
r			LDR		R4,R1,#0		;
			NOT		R2,R4			;
			ADD		R2,R2,#1		;
			ADD		R2,R2,R0		;
			BRz		right2			;
			ADD		R1,R1,#1		;
			ADD		R5,R5,#1		;
			LD		R2,Eighteen		;
			ADD   		R2,R5,R2		;
			BRz		left2			;
			BRnp		r			;
 
 
right2			AND		R5,R5,#0		;
			LD		R4,FOUR			;
			LD		R3,SR0			;
			LD		R1,NFOUR		;
 
Heng1			LDR		R2,R3,#0		;
			BRn		Add3			;
			BRzp		Finish2			;
 
 
Add3			ADD		R5,R5,#1		;
			ADD		R2,R5,R1		;
			BRz		WIN2			;
			ADD		R4,R4,#-1		;
			BRz		Finish2			;
			ADD		R3,R3,#1		;
			BRp		Heng1			;
 
 
 
left2			AND		R5,R5,#0		;
			LD		R4,FOUR			;
			LD		R3,SR0			;
			LD		R1,NFOUR		;
 
HENG4			LDR		R2,R3,#0		;
			BRn		Add4			;
			BRzp		Finish2			;
 
 
Add4			ADD		R5,R5,#1		;
			ADD		R2,R5,R1		;
			BRz		WIN2			;
			ADD		R4,R4,#-1		;
			BRz		Finish2			;
			ADD		R3,R3,#-1		;
			BRp		HENG4			;
 
 
Win2			LEA		R0,U2WIN		;
			PUTS					;
			ADD		R6,R6,#1		;
			BRnzp		Finish2			;
	
 
 
Finish2			LD		R1,SR1			;
			LD		R2,SR2			;
			LD		R3,SR3			;
			LD 		R4,SR4			;
			LD		R7,SR7			;
			LD		R0,SR0			;
			LD		R5,SR5			;
			RET
 
 
RIGHT			.fill		x3500			;
			.fill		x3501			;
			.fill		x3502			;
			.fill		x3506			;
			.fill		x3507			;
			.fill		x3508			;
			.fill		x350C			;
			.fill		x350D			;
			.fill		x350E			;
			.fill		x3512			;
			.fill		x3513			;
			.fill		x3514			;
			.fill		x3518			;
			.fill		X3519			;
			.fill		X351A			;
			.fill		x351E			;
			.fill		x351F			;
			.fill		x3520			;
 
 
Eighteen		.fill		#-18			;	
 
 
;判断是否用户1或用户2是否赢了比赛（斜1）
IsXie1			ST		R1,S1			;	
			ST		R2,S2			;
			ST		R3,S3			;
			ST 		R4,S4			;
			ST		R7,S7			;
			ST		R0,S0			;
			ST		R5,S5			;
			ST		R6,S6			;
		
			AND		R5,R5,#0		;
			AND		R6,R6,#0		;
			
			LEA		R1,XIEDATA		;	
X			LDR		R4,R1,#0		;
			NOT		R2,R4			;
			ADD		R2,R2,#1		;
			ADD		R2,R2,R0		;
			BRz		XIE			;
			ADD		R1,R1,#1		;
			ADD		R5,R5,#1		;		
			ADD   		R2,R5,#-9		;
			BRz		FinishXIE1		;
			BRnp		X			;
 
 
XIE			LD		R2,S6			;
			BRp		XIE1			;
			BRn		XIE2			;
 
XIE1			AND		R5,R5,#0		;
			LD		R4,FOUR			;
			LD		R3,S0			;
			LD		R1,NFOUR		;
 
xie1			LDR		R2,R3,#0		;
			BRp		jia1			;
			BRnz		FinishXIE1		;
 
 
jia1			ADD		R5,R5,#1		;
			ADD		R2,R5,R1		;
			BRz		win1			;
			ADD		R4,R4,#-1		;
			BRz		FinishXIE1		;
			ADD		R3,R3,#7		;
			BRp		xie1			;
 
 
XIE2			AND		R5,R5,#0		;
			LD		R4,FOUR			;
			LD		R3,SR0			;
			LD		R1,NFOUR		;
 
 
xie2			LDR		R2,R3,#0		;
			BRn		jia2			;
			BRzp		FinishXIE1		;
 
 
jia2			ADD		R5,R5,#1		;
			ADD		R2,R5,R1		;
			BRz		win2			;
			ADD		R4,R4,#-1		;
			BRz		FinishXIE1		;
			ADD		R3,R3,#7		;
			BRp		xie2			;
 
win1			ADD		R6,R6,#1		;
			LEA		R0,U1WIN		;
			PUTS					;
			BRnzp		FinishXIE1		;
 
win2			ADD		R6,R6,#1		;
			LEA		R0,U2WIN		;
			PUTS					;
			BRnzp		FinishXIE1		;
			
FinishXIE1		LD		R1,S1			;
			LD		R2,S2			;
			LD		R3,S3			;
			LD 		R4,S4			;
			LD		R7,S7			;
			LD		R0,S0			;
			LD		R5,S5			;
			RET
 
XIEDATA			.fill		x3500			;
			.fill		x3501			;
			.fill		x3502			;
			.fill		x3506			;
			.fill		x3507			;
			.fill		x3508			;
			.fill		x350C			;
			.fill		x350D			;
			.fill		x350E			;
 
			
;判断是否用户1或用户2是否赢了比赛（斜2）
IsXie2			ST		R1,S1			;	
			ST		R2,S2			;
			ST		R3,S3			;
			ST 		R4,S4			;
			ST		R7,S7			;
			ST		R0,S0			;
			ST		R5,S5			;
			ST		R6,S6			;
		
			AND		R5,R5,#0		;R5作为判断四个点中有多少个符合条件的点的计数器
			AND		R6,R6,#0		;R6作为是否赢了的状态码，0则没赢，1则赢
			
			LEA		R1,XIEDATA2		;R1指向要往左上寻找的数组的首地址	
XX			LDR		R4,R1,#0		;
			NOT		R2,R4			;
			ADD		R2,R2,#1		;
			ADD		R2,R2,R0		;
			BRz		xie			;
			ADD		R1,R1,#1		;
			ADD		R5,R5,#1		;
			
			ADD   		R2,R5,#-9		;
			BRz		Finishxie1		;
			BRnp		XX			;
 
xie			LD		R2,S6			;
			BRp		xiE1			;
			BRn		xiE2			;
 
xiE1			AND		R5,R5,#0		;
			LD		R4,Four			;
			LD		R3,S0			;
			LD		R1,NFour		;
 
xie11			LDR		R2,R3,#0		;
			BRp		Jia1			;
			BRnz		FinishXIE1		;
 
 
Jia1			ADD		R5,R5,#1		;
			ADD		R2,R5,R1		;
			BRz		wiN1			;
			ADD		R4,R4,#-1		;
			BRz		Finishxie1		;
			ADD		R3,R3,#-7		;
			BRp		xie11			;
 
 
xiE2			AND		R5,R5,#0		;
			LD		R4,Four			;
			LD		R3,S0			;
			LD		R1,NFour		;
 
 
xie22			LDR		R2,R3,#0		;
			BRn		Jia2			;
			BRzp		FinishXIE1		;
 
 
Jia2			ADD		R5,R5,#1		;
			ADD		R2,R5,R1		;
			BRz		wiN2			;
			ADD		R4,R4,#-1		;
			BRz		Finishxie1		;
			ADD		R3,R3,#-7		;
			BRp		xie22			;
 
wiN1			ADD		R6,R6,#1		;
			LEA		R0,U1win		;
			PUTS					;
			BRnzp		Finishxie1		;
 
wiN2			ADD		R6,R6,#1		;
			LEA		R0,U2win		;
			PUTS					;
			BRnzp		Finishxie1		;
			
Finishxie1		LD		R1,S1			;
			LD		R2,S2			;
			LD		R3,S3			;
			LD 		R4,S4			;
			LD		R7,S7			;
			LD		R0,S0			;
			LD		R5,S5			;
			RET
 
 
;该数组是棋盘右下角，要往左上查询四个点
XIEDATA2		.fill		x3515			;
			.fill		x3516			;
			.fill		x3117			;
			.fill		x351B			;
			.fill		x351C			;
			.fill		x351D			;
			.fill		x3521			;
			.fill		x3522			;
			.fill		x3523			;
Four			.fill		x0004			;
NFour			.fill		#-4			;	
U1win			.stringz	"Player 1 Wins."	;
U2win			.stringz	"Player 2 Wins."	;
S0			.BLKW		1			;
S1			.BLKW		1			;
S2			.BLKW		1			;
S3			.BLKW		1			;
S4			.BLKW		1			;
S5			.BLKW		1			;
S6			.BLKW		1			;
S7			.BLKW		1			;
 
 
 
;判断是否用户1或用户2是否赢了比赛（斜3）
IsXie3			ST		R1,S1			;
			ST		R2,S2			;
			ST		R3,S3			;
			ST 		R4,S4			;
			ST		R7,S7			;
			ST		R0,S0			;
			ST		R5,S5			;
			ST		R6,S6			;
		
			AND		R5,R5,#0		;
			AND		R6,R6,#0		;
 
			LEA		R1,XIEDATA3		;	
xxx			LDR		R4,R1,#0		;
			NOT		R2,R4			;
			ADD		R2,R2,#1		;
			ADD		R2,R2,R0		;
			BRz		xiee			;
			ADD		R1,R1,#1		;
			ADD		R5,R5,#1		;
			ADD		R2,R5,#-9		;
			BRz		FinishXIE3		;
			BRnp		xxx			;
 
 
xiee			LD		R2,S6			;
			BRp		Xiee1			;
			BRn		Xiee2			;
 
 
Xiee1			AND		R5,R5,#0		;
			LD		R4,Four			;
			LD		R3,S0			;
			LD		R1,NFour		;
 
xie111			LDR		R2,R3,#0		;
			BRp		JIa1			;
			BRnz		FinishXIE3		;
 
JIa1			ADD		R5,R5,#1		;
			ADD		R2,R5,R1		;
			BRz		WIn1			;
			ADD		R4,R4,#-1		;
			BRz		FinishXIE3		;
			ADD		R3,R3,#5		;
			BRp		xie111			;
 
 
Xiee2			AND		R5,R5,#0		;
			LD		R4,Four			;
			LD		R3,S0			;
			LD		R1,NFour		;
 
xie222			LDR		R2,R3,#0		;
			BRn		JIa2			;
			BRzp		FinishXIE3		;
 
 
JIa2			ADD		R5,R5,#1		;
			ADD		R2,R5,R1		;
			BRz		WIn2			;
			ADD		R4,R4,#-1		;
			BRz		FinishXIE3		;
			ADD		R3,R3,#5		;
			BRp		xie222			;
 
WIn1			ADD		R6,R6,#1		;
			LEA		R0,U1win		;
			PUTS					;
			BRnzp		FinishXIE3		;
 
WIn2			ADD		R6,R6,#1		;
			LEA		R0,U2win		;
			PUTS					;
			BRnzp		FinishXIE3		;
 
	
FinishXIE3		LD		R1,S1			;
			LD		R2,S2			;
			LD		R3,S3			;
			LD 		R4,S4			;
			LD		R7,S7			;
			LD		R0,S0			;
			LD		R5,S5			;
			RET
 
;该数组是棋盘右上角，要往左下查询四个点			
XIEDATA3		.fill		x3503			;
			.fill		x3504			;
			.fill		x3505			;
			.fill		x3509			;
			.fill		x350A			;
			.fill		x350B			;
			.fill		x350F			;
			.fill		x3510			;
			.fill		x3511			;
 
 
 
;判断是否用户1或用户2是否赢了比赛（斜4）
IsXie4			ST		R1,S1			;
			ST		R2,S2			;
			ST		R3,S3			;
			ST 		R4,S4			;
			ST		R7,S7			;
			ST		R0,S0			;
			ST		R5,S5			;
			ST		R6,S6			;
		
			AND		R5,R5,#0		;
			AND		R6,R6,#0		;
 
			LEA		R1,XIEDATA4		;	
xxxx			LDR		R4,R1,#0		;
			NOT		R2,R4			;
			ADD		R2,R2,#1		;
			ADD		R2,R2,R0		;
			BRz		xiee1			;
			ADD		R1,R1,#1		;
			ADD		R5,R5,#1		;
			ADD		R2,R5,#-9		;
			BRz		FinishXIE4		;
			BRnp		xxxx			;
 
 
xiee1			LD		R2,S6			;
			BRp		Xiee11			;
			BRn		Xiee22			;
 
 
Xiee11			AND		R5,R5,#0		;
			LD		R4,Four			;
			LD		R3,S0			;
			LD		R1,NFour		;
 
xie1111			LDR		R2,R3,#0		;
			BRp		JIa11			;
			BRnz		FinishXIE4		;
 
JIa11			ADD		R5,R5,#1		;
			ADD		R2,R5,R1		;
			BRz		WIn11			;
			ADD		R4,R4,#-1		;
			BRz		FinishXIE4		;
			ADD		R3,R3,#-5		;
			BRp		xie1111			;
 
Xiee22			AND		R5,R5,#0		;
			LD		R4,Four			;
			LD		R3,S0			;
			LD		R1,NFour		;
 
xie2222			LDR		R2,R3,#0		;
			BRn		JIa22			;
			BRzp		FinishXIE4		;
 
JIa22			ADD		R5,R5,#1		;
			ADD		R2,R5,R1		;
			BRz		WIn22			;
			ADD		R4,R4,#-1		;
			BRz		FinishXIE4		;
			ADD		R3,R3,#-5		;
			BRp		xie2222			;
 
WIn11			ADD		R6,R6,#1		;
			LEA		R0,U1win		;
			PUTS					;
			BRnzp		FinishXIE4		;
 
WIn22			ADD		R6,R6,#1		;
			LEA		R0,U2win		;
			PUTS					;
			BRnzp		FinishXIE4		;
 
	
FinishXIE4		LD		R1,S1			;
			LD		R2,S2			;
			LD		R3,S3			;
			LD 		R4,S4			;
			LD		R7,S7			;
			LD		R0,S0			;
			LD		R5,S5			;
			RET
 
;该数组是棋盘左下角，要往右上查询四个点			
XIEDATA4		.fill		x3512			;
			.fill		x3513			;
			.fill		x3514			;
			.fill		x3518			;
			.fill		x3519			;
			.fill		x351A			;
			.fill		x351E			;
			.fill		x351F			;
			.fill		x3520			;
 
			.END