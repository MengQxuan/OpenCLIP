import open_clip
import inspect

import open_clip.model as m
import open_clip.transformer as t
import open_clip.tokenizer as tok
import open_clip.factory as f

print("factory.py:", f.__file__)
print("model.py  :", m.__file__)
print("transformer.py:", t.__file__)
print("tokenizer.py:", tok.__file__)

# 打印关键函数所在文件与源码行号范围（定位用）
for fn in [f.create_model_and_transforms, f.create_model]:
    print(f"\n{fn.__name__} defined in:", fn.__code__.co_filename, "line", fn.__code__.co_firstlineno)

# CLIP 类位置
print("\nCLIP class defined in:", inspect.getsourcefile(m.CLIP))
