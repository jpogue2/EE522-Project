import slab

hrtf = slab.HRTF.kemar()
sound = slab.Sound.whitenoise(duration=1.0)

# 0 = front
front = hrtf.apply(0, sound)
front.play()

# 90 = left
left = hrtf.apply(90, sound)
left.play()