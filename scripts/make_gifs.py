from PIL import Image

images = []
for i in range(9):
    images.append(Image.open(f"../examples/lf/results/render_0{i}_{i}.0.png"))
images[0].save("../examples/lf/out.gif", save_all=True, append_images=images[1:], duration=100, loop=0)