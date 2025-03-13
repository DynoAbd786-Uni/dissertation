import warp as wp

C = 3

@wp.kernel
def k():
    print(C)

wp.launch(k, dim=1)

# redefine constant
C = 42

# tell Warp that the module was modified
k.module.mark_modified()

wp.launch(k, dim=1)