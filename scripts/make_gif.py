import imageio.v2 as imageio
from PIL import Image
import os

# 圖片順序（你可以自由改）
image_paths = [
    "docs/cagr_bootstrap_3y.png",
    "docs/cagr_bootstrap_5y.png",
    "docs/cagr_bootstrap_10y.png",
    "docs/cagr_bootstrap_20y.png",
    "docs/cagr_bootstrap_30y.png"

]

images = []


images = [Image.open(p) for p in image_paths]
durations = [2500] + [1200] * (len(images) - 2) + [2500]  # 每張圖片的停留時間（毫秒）

images[0].save(
    "docs/cagr_bootstrap_animation.gif",
    save_all=True,
    append_images=images[1:],
    duration=durations,  # 毫秒（1.5秒）
    loop=0
)


print("GIF created!")


image_paths = [
    "docs/cagr_gbm_3y.png",
    "docs/cagr_gbm_5y.png",
    "docs/cagr_gbm_10y.png",
    "docs/cagr_gbm_20y.png",
    "docs/cagr_gbm_30y.png"

]

images = []


images = [Image.open(p) for p in image_paths]
durations = [2500] + [1200] * (len(images) - 2) + [2500]  # 每張圖片的停留時間（毫秒）

images[0].save(
    "docs/cagr_gbm_animation.gif",
    save_all=True,
    append_images=images[1:],
    duration=durations,  # 毫秒（1.5秒）
    loop=0
)


print("GIF created!")
