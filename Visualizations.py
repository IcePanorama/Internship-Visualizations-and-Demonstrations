import matplotlib.pyplot as plt
import numpy as np


def generate_background() -> None:
    labels = ["P.O.", "S.M.", "E.T."]
    
    plt.style.use('_mpl-gallery')
    
    fig, ax = plt.subplots()
    
    ax.set(xlim=(0,4), xticks=np.arange(1,4),
           ylim=(0,4), yticks=np.arange(1, 4))
    
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title("The Intersections of Fairness")
    
    plt.show()
    
    
def generate_labels(PREFIXES : list[str], SUFFIXES : list[str]) -> list[str]:
    arr = []
    
    for i in PREFIXES:
        for j in SUFFIXES:
            arr.append(i + "+" + j)
            
    return arr


def generate_pie_chart(DATA : list[int], LABELS : list[str]):
    plt.style.use('_mpl-gallery-nogrid')
    
    colors = plt.get_cmap('Blues')(np.linspace(0.2, 0.7, len(po_et)))
    
    fig, ax = plt.subplots()
    
    ax.pie(DATA, labels=LABELS, colors=colors, radius=3, center=(4,4), 
               wedgeprops={"linewidth": 1, "edgecolor": "white"}, frame=False)

    ax.set(xlim=(0,8), ylim=(0,8))
    
    plt.show()


if __name__ == "__main__":    
    #generate_background()
    
    po = ["I", "B", "G"]
    et = ["Pr", "In", "Po"]
    sm = ["WB", "GB", "BB"]
    
    #see below for raw data/meaning
    po_et = [1, 1, 1, 7, 6, 10, 4, 8, 5]
    po_sm = [1, 1, 0, 2, 1, 6, 5, 5, 2]
    et_sm = [1, 3, 5, 5, 3, 4, 3, 2, 8]
    
    #generate_pie_chart(po_et, generate_labels(po, et))
    #generate_pie_chart(po_sm, generate_labels(po, sm))
    #generate_pie_chart(et_sm, generate_labels(et, sm))

'''
### RAW DATA ###
PO & ET:
Indi+Pre = 1
Indie+In = 1
Indie+Post = 1

B+Pr = 7
B+In = 6
B+Po = 10

Group+Pre = 4
Gr+In = 8
Gr+Po = 5

##
PO & SM:
I+WB = 1
I+GB = 1
I+BB = 0

B+WB = 2
B+GB = 1
B+BB = 6

G+WB = 5
G+GB = 5
G+BB = 2

##
ET & SM:
Pr+WB = 1
Pr+GB = 3
Pr+BB = 5

In+WB = 5
In+GB = 3
In+BB = 4

Po+WB = 3
Po+GB = 2
Po+BB = 8
'''