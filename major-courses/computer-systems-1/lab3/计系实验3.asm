.ORIG x3000
; 初始化指针和计数器
        LEA R0, Score           ; R0 -> Score base
        LEA R1, Sorted_score    ; R1 -> Sorted_score base
        AND R2, R2, #0          ; R2 = i = 0

; -------- 拷贝成绩到 Sorted_score --------
CopyLoop
        LDR R3, R0, #0
        STR R3, R1, #0
        ADD R0, R0, #1
        ADD R1, R1, #1
        ADD R2, R2, #1
        ADD R4, R2, #-16
        BRn CopyLoop            ; Repeat until 16 done

; -------- 冒泡排序 Sorted_score 中的成绩 --------
        LEA R1, Sorted_score
        AND R6, R6, #0          ; Outer loop i = 0
OuterLoop
        AND R7, R7, #0          ; Inner loop j = 0
InnerLoop
        ADD R3, R7, #1
        ADD R4, R1, R7
        LDR R5, R4, #0          ; score[j]
        ADD R4, R1, R3
        LDR R0, R4, #0          ; score[j+1]
        NOT R4, R0
        ADD R4, R4, #1
        ADD R4, R5, R4          ; R4 = score[j] - score[j+1]
        BRp NoSwap
        ; Swap
        ADD R2, R1, R7
	STR R0, R2, #0
        ADD R2, R1, R3
	STR R5, R2, #0
NoSwap
        ADD R7, R7, #1
        ADD R5, R7, #-15
        BRn InnerLoop
        ADD R6, R6, #1
        ADD R5, R6, #-15
        BRn OuterLoop

; -------- 评分 --------
        AND R3, R3, #0          ; Grade_A
        AND R4, R4, #0          ; Grade_B
        AND R5, R5, #0          ; Grade_C
        LEA R1, Sorted_score
        AND R2, R2, #0          ; index i = 0

GradeLoop
        LDR R0, R1, #0          ; 当前成绩
        ADD R1, R1, #1

        ; 判断 A: i < 4 且 score >= 85
        ADD R6, R2, #-4
        BRzp NotA               ; i >= 4 -> Not A
        LD R7, CONST_85
        NOT R7, R7
        ADD R7, R7, #1
        ADD R6, R0, R7          ; score - 85
        BRn NotA                ; score < 85 -> Not A
        ADD R3, R3, #1          ; Grade_A++
        BR NextGrade

NotA
        ; 判断 B: i < 8 且 score >= 75
        ADD R6, R2, #-8
        BRzp NotB               ; i >= 8 -> Not B
        LD R7, CONST_75
        NOT R7, R7
        ADD R7, R7, #1
        ADD R6, R0, R7          ; score - 75
        BRn NotB
        ADD R4, R4, #1          ; Grade_B++
        BR NextGrade

NotB
        ADD R5, R5, #1          ; Grade_C++

NextGrade
        ADD R2, R2, #1
        ADD R6, R2, #-16
        BRn GradeLoop

; 存储结果
    ST R3, Grade_A
    ST R4, Grade_B
    ST R5, Grade_C
	AND R0, R0, #0
	AND R1, R1, #0
	LEA R0, MESGA
	PUTS
	LD R0, Grade_A
	LD R1, CONST_ASCII
	ADD R0, R1, R0
	OUT
	LD R0, CONST_ENTER
	OUT
	LEA R0, MESGB
	PUTS
	LD R0, Grade_B
	ADD R0, R1, R0
	OUT
	LD R0, CONST_ENTER
	OUT
	LEA R0, MESGC
	PUTS
	LD R0, Grade_C
	ADD R0, R1, R0	
	OUT
	LD R0, CONST_ENTER
	OUT
        HALT

; -------- DATA --------
MESGA .STRINGZ "GradeA " 
MESGB .STRINGZ "GradeB "
MESGC .STRINGZ "GradeC " 
CONST_85 .FILL #85
CONST_75 .FILL #75
CONST_ASCII .FILL X30
CONST_ENTER .FILL Xa
Grade_A .BLKW 1
Grade_B .BLKW 1
Grade_C .BLKW 1
Sorted_score .BLKW 16
Score   .FILL #99
        .FILL #56
        .FILL #78
        .FILL #79
        .FILL #90
        .FILL #67
        .FILL #95
        .FILL #82
        .FILL #95
        .FILL #76
        .FILL #69
        .FILL #83
        .FILL #75
        .FILL #87
        .FILL #82
        .FILL #23
.END
HALT
