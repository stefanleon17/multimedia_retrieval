from obj_viewer import display

def main():
    #filename = r"ShapeDatabase/Car/m1487.obj"
    filename = r"ShapeDatabase/Door/D01121.obj"
    #filename = r"ShapeDatabase/PlantIndoors/D00159.obj"
    width, height = 900, 700
    display(filename, width, height)


if __name__ == '__main__':
       main()