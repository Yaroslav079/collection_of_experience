def solve(ind, expr):
    seq = "9876543210"
    
    if ind == len(seq) - 1:
        res, num, sign = 0, "", "+"
        for char in expr + "+":
            if char in "+-":
                if sign == "+":
                    res += int(num)
                else:
                    res -= int(num)
                num, sign = "", char
            else:
                num += char
        if res == 200:
            print(expr + "=200")
        return

    for sign in ["+", "-", ""]:
        solve(ind + 1, expr + sign + seq[ind + 1])


solve(0, "9")

"""
9-8+7-6-5-4-3+210=200
9-8-7-6-5+4+3+210=200
98+76-5+43-2-10=200
98-7+65+43+2-1+0=200
98-7+65+43+2-1-0=200
"""
