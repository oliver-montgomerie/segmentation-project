
print("Hello")

a = 1
b = 2
c = 3

with open('info.txt','w') as f:
        txt = ["Test set mean dice:", a,
                "\nNumber of train files:", b,
                "\nNumber of val files:", c,
        ]
        for t in txt:
            f.write(f"{t}")