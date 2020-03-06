def dnn_to_tikz(layer_sizes, fname):
    with open(fname, 'w') as f:

        # Write tikz header
        f.write('\\begin{tikzpicture}\n')
    
        n_layers = len(layer_sizes)

        for j, s in enumerate(layer_sizes):
            if s <= 16:
                for i in range(s):
                    f.write(f'\\node[circle, draw, minimum size=0.5cm] at ({j},{i}) ({j}{i})')
                    f.write(' {};\n')
            else:
                for i in range(7):
                    f.write(f'\\node[circle, draw, minimum size=0.5cm] at ({j},{i}) ({j}{i})')
                    f.write(' {};\n')
                f.write(f'\\node[] at ({j},{7}) ({j}7)')
                f.write(' {\\vdots};\n')
                for i in range(8,16):
                    f.write(f'\\node[circle, draw, minimum size=0.5cm] at ({j},{i}) ({j}{i})')
                    f.write(' {};\n')
        
        f.write('\\end{tikzpicture}\n')

if __name__ == '__main__':
    layer_sizes = [1601*8, 1024, 512, 256, 128, 64, 32, 16, 8, 8]
    dnn_to_tikz(layer_sizes, 'dnn.tikz')
