from global_config import *

class small_gemm_nn_def:
    def load_a0b0(a0, pga, b0, pgb, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"{ldopt}      {b0}.s, {pgb}, [{pB0}, z27.s, UXTW]\n"
        code_str += f"{ldaopt}     {a0}.s, {pga}, [{pA0}]\n"

        return code_str

    def load_a1(a1, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"{ldaopt}     {a1}.s, {pg}, [{pA0}, #1, MUL VL]\n"

        return code_str

    def load_a2(a2, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"{ldaopt}     {a2}.s, {pg}, [{pA0}, #2, MUL VL]\n"

        return code_str

    def load_a3(a3, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"{ldaopt}     {a3}.s, {pg}, [{pA0}, #3, MUL VL]\n"

        return code_str

    def load_b1(b1, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pBn}, {pB0}, {pB_OFFSET}\n"
        code_str += f"{ldopt}      {b1}.s, {pg}, [{pBn}, z27.s, UXTW]\n"

        return code_str

    def load_b2(b2, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pBn}, {pB0}, {pB_OFFSET}, lsl #1\n"
        code_str += f"{ldopt}      {b2}.s, {pg}, [{pBn}, z27.s, UXTW]\n"

        return code_str

    def load_b3(b3, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pBn}, {pB0}, {pB_OFFSET}\n"
        code_str += f"add          {pBn}, {pBn}, {pB_OFFSET}, lsl #1\n"
        code_str += f"{ldopt}      {b3}.s, {pg}, [{pBn}, z27.s, UXTW]\n"

        return code_str

    def set_svindex():
        # nopackA nopackB
        code_str = f""
        code_str += f"lsl     {TMP_CNT}, {LDB}, #2\n"
        code_str += f"mov     z27.s, #0\n"
        code_str += f"index   z27.s, #0, {TMP_CNT_SIN}\n"

        return code_str

    def kernel_mm_loop_n_pre_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"mov     {pAt}, {origPA}\n"
        code_str += f"mov     {pB0}, {pBt}\n"

        return code_str

    def kernel_mm_loop_n_post_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"mul     {TMP_CNT}, {LDB}, {MIN_N}\n"
        code_str += f"add     {pBt}, {pBt}, {TMP_CNT}, lsl #2\n"

        return code_str

    def kernel_mm_loop_m_pre_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"mov      {pB0}, {pBt}      // KERNEL_MM_LOOP_M_PRE_FUNC\n"
        code_str += f"mov      {pA0}, {pAt}\n"
        code_str += f"lsl      {pB_OFFSET}, {LDB}, #6\n"
        code_str += f"mov      {OFFSET_A}, {LDA}\n"
        code_str += f"mov      {OFFSET_B}, #1\n"

        return code_str

    def kernel_mm_loop_m_post_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"add      {pAt}, {pAt}, {MIN_M}, lsl #2\n"

        return code_str

class small_gemm_nt_def:
    def load_a0b0(a0, pga, b0, pgb, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"{ldopt}      {b0}.s, {pgb}, [{pB0}]\n"
        code_str += f"{ldaopt}     {a0}.s, {pga}, [{pA0}]\n"

        return code_str

    def load_a1(a1, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"{ldaopt}     {a1}.s, {pg}, [{pA0}, #1, MUL VL]\n"

        return code_str

    def load_a2(a2, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"{ldaopt}     {a2}.s, {pg}, [{pA0}, #2, MUL VL]\n"

        return code_str

    def load_a3(a3, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"{ldaopt}     {a3}.s, {pg}, [{pA0}, #3, MUL VL]\n"

        return code_str

    def load_b1(b1, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"{ldopt}      {b1}.s, {pg}, [{pB0}, #1, MUL VL]\n"

        return code_str

    def load_b2(b2, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"{ldopt}      {b2}.s, {pg}, [{pB0}, #2, MUL VL]\n"

        return code_str

    def load_b3(b3, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"{ldopt}      {b3}.s, {pg}, [{pB0}, #3, MUL VL]\n"

        return code_str

    def set_svindex():
        # nopackA nopackB
        code_str = f""

        return code_str

    def kernel_mm_loop_n_pre_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"mov     {pAt}, {origPA}\n"
        code_str += f"mov     {pB0}, {pBt}\n"

        return code_str

    def kernel_mm_loop_n_post_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"add     {pBt}, {pBt}, {MIN_N}, lsl #2\n"

        return code_str

    def kernel_mm_loop_m_pre_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"mov      {pB0}, {pBt}\n"
        code_str += f"mov      {pA0}, {pAt}\n"
        code_str += f"mov      {OFFSET_A}, {LDA}\n"
        code_str += f"mov      {OFFSET_B}, {LDB}\n"

        return code_str

    def kernel_mm_loop_m_post_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"add      {pAt}, {pAt}, {MIN_M}, lsl #2\n"

        return code_str

class small_gemm_tn_def:
    def load_a0b0(a0, pga, b0, pgb, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"{ldopt}      {b0}.s, {pgb}, [{pB0}, z27.s, UXTW]\n"
        code_str += f"{ldopt}      {a0}.s, {pga}, [{pA0}, z28.s, UXTW]\n"

        return code_str

    def load_a1(a1, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pAn}, {pA0}, {pA_OFFSET}\n"
        code_str += f"{ldopt}      {a1}.s, {pg}, [{pAn}, z28.s, UXTW]\n"

        return code_str

    def load_a2(a2, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pAn}, {pA0}, {pA_OFFSET}, lsl #1\n"
        code_str += f"{ldopt}      {a2}.s, {pg}, [{pAn}, z28.s, UXTW]\n"

        return code_str

    def load_a3(a3, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pAn}, {pA0}, {pA_OFFSET}\n"
        code_str += f"add          {pAn}, {pAn}, {pA_OFFSET}, lsl #1\n"
        code_str += f"{ldopt}      {a3}.s, {pg}, [{pAn}, z28.s, UXTW]\n"

        return code_str

    def load_b1(b1, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pBn}, {pB0}, {pB_OFFSET}\n"
        code_str += f"{ldopt}      {b1}.s, {pg}, [{pBn}, z27.s, UXTW]\n"

        return code_str

    def load_b2(b2, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pBn}, {pB0}, {pB_OFFSET}, lsl #1\n"
        code_str += f"{ldopt}      {b2}.s, {pg}, [{pBn}, z27.s, UXTW]\n"

        return code_str

    def load_b3(b3, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pBn}, {pB0}, {pB_OFFSET}\n"
        code_str += f"add          {pBn}, {pBn}, {pB_OFFSET}, lsl #1\n"
        code_str += f"{ldopt}      {b3}.s, {pg}, [{pBn}, z27.s, UXTW]\n"

        return code_str

    def set_svindex():
        # nopackA nopackB
        code_str = f""
        code_str += f"lsl     {TMP_CNT}, {LDB}, #2\n"
        code_str += f"mov     z27.s, #0\n"
        code_str += f"index   z27.s, #0, {TMP_CNT_SIN}\n"
        code_str += f"lsl     {TMP_CNT}, {LDA}, #2\n"
        code_str += f"mov     z28.s, #0\n"
        code_str += f"index   z28.s, #0, {TMP_CNT_SIN}\n"

        return code_str

    def kernel_mm_loop_n_pre_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"mov     {pAt}, {origPA}\n"
        code_str += f"mov     {pB0}, {pBt}\n"

        return code_str

    def kernel_mm_loop_n_post_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"mul     {TMP_CNT}, {LDB}, {MIN_N}\n"
        code_str += f"add     {pBt}, {pBt}, {TMP_CNT}, lsl #2\n"

        return code_str

    def kernel_mm_loop_m_pre_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"mov      {pB0}, {pBt}\n"
        code_str += f"mov      {pA0}, {pAt}\n"
        code_str += f"lsl      {pA_OFFSET}, {LDA}, #6\n"
        code_str += f"lsl      {pB_OFFSET}, {LDB}, #6\n"
        code_str += f"mov      {OFFSET_A}, #1\n"
        code_str += f"mov      {OFFSET_B}, #1\n"

        return code_str

    def kernel_mm_loop_m_post_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"mul     {TMP_CNT}, {LDA}, {MIN_M}\n"
        code_str += f"add      {pAt}, {pAt}, {TMP_CNT}, lsl #2\n"

        return code_str

class small_gemm_tt_def:
    def load_a0b0(a0, pga, b0, pgb, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"{ldopt}      {b0}.s, {pgb}, [{pB0}]\n"
        code_str += f"{ldopt}      {a0}.s, {pga}, [{pA0}, z28.s, UXTW]\n"

        return code_str

    def load_a1(a1, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pAn}, {pA0}, {pA_OFFSET}\n"
        code_str += f"{ldopt}      {a1}.s, {pg}, [{pAn}, z28.s, UXTW]\n"

        return code_str

    def load_a2(a2, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pAn}, {pA0}, {pA_OFFSET}\n"
        code_str += f"add          {pAn}, {pAn}, {pA_OFFSET}\n"
        code_str += f"{ldopt}      {a2}.s, {pg}, [{pAn}, z28.s, UXTW]\n"

        return code_str

    def load_a3(a3, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pAn}, {pA0}, {pA_OFFSET}\n"
        code_str += f"add          {pAn}, {pAn}, {pA_OFFSET}\n"
        code_str += f"add          {pAn}, {pAn}, {pA_OFFSET}\n"
        code_str += f"{ldopt}      {a3}.s, {pg}, [{pAn}, z28.s, UXTW]\n"

        return code_str

    def load_b1(b1, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"{ldopt}      {b1}.s, {pg}, [{pB0}, #1, MUL VL]\n"

        return code_str

    def load_b2(b2, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"{ldopt}      {b2}.s, {pg}, [{pB0}, #2, MUL VL]\n"

        return code_str

    def load_b3(b3, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"{ldopt}      {b3}.s, {pg}, [{pB0}, #3, MUL VL]\n"

        return code_str

    def set_svindex():
        # nopackA nopackB
        code_str = f""
        code_str += f"lsl     {TMP_CNT}, {LDA}, #2\n"
        code_str += f"mov     z28.s, #0\n"
        code_str += f"index   z28.s, #0, {TMP_CNT_SIN}\n"

        return code_str

    def kernel_mm_loop_n_pre_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"mov     {pAt}, {origPA}\n"
        code_str += f"mov     {pB0}, {pBt}\n"

        return code_str

    def kernel_mm_loop_n_post_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"add     {pBt}, {pBt}, {MIN_N}, lsl #2\n"

        return code_str

    def kernel_mm_loop_m_pre_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"mov      {pB0}, {pBt}\n"
        code_str += f"mov      {pA0}, {pAt}\n"
        code_str += f"lsl      {pA_OFFSET}, {LDA}, #6\n"
        code_str += f"mov      {OFFSET_A}, #1\n"
        code_str += f"mov      {OFFSET_B}, {LDB}\n"

        return code_str

    def kernel_mm_loop_m_post_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"mul     {TMP_CNT}, {LDA}, {MIN_M}\n"
        code_str += f"add      {pAt}, {pAt}, {TMP_CNT}, lsl #2\n"

        return code_str

class general_gemm_def:
    def load_a0b0(a0, pga, b0, pgb, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"{ldopt}      {b0}.s, {pgb}, [{pB0}]\n"
        code_str += f"{ldaopt}     {a0}.s, {pga}, [{pA0}]\n"

        return code_str

    def load_a1(a1, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"{ldaopt}     {a1}.s, {pg}, [{pA0}, #1, MUL VL]\n"

        return code_str

    def load_a2(a2, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"{ldaopt}     {a2}.s, {pg}, [{pA0}, #2, MUL VL]\n"

        return code_str

    def load_a3(a3, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"{ldaopt}     {a3}.s, {pg}, [{pA0}, #3, MUL VL]\n"

        return code_str

    def load_b1(b1, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"{ldopt}      {b1}.s, {pg}, [{pB0}, #1, MUL VL]\n"

        return code_str

    def load_b2(b2, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"{ldopt}      {b2}.s, {pg}, [{pB0}, #2, MUL VL]\n"

        return code_str

    def load_b3(b3, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"{ldopt}      {b3}.s, {pg}, [{pB0}, #3, MUL VL]\n"

        return code_str

    def set_svindex():
        # nopackA nopackB
        code_str = f""

        return code_str

    def kernel_mm_loop_n_pre_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"mov     {pAt}, {origPA}\n"
        code_str += f"mov     {pB0}, {pBt}\n"

        return code_str

    def kernel_mm_loop_n_post_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"mov     {pBt}, {pB0}\n"

        return code_str

    def kernel_mm_loop_m_pre_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"mov      {pB0}, {pBt}\n"
        code_str += f"mov      {pA0}, {pAt}\n"

        return code_str

    def kernel_mm_loop_m_post_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"add      {pAt}, {pA0}\n"

        return code_str


