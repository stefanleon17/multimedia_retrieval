from obj_viewer import display

def main():
    #filename = r"ResampledShapeDatabase/Apartment/D00478.obj"
    #filename = r"ResampledShapeDatabase/Tree/m1070.obj"
    #filename = r"ResampledShapeDatabase/Cellphone/D00327.obj"

    filename = r"TestShapesResampled/half_sphere_dented.obj"
    width, height = 900, 700
    display(filename, width, height)


if __name__ == '__main__':
       main()
