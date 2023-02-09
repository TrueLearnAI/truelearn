from truelearn.preprocessing.wikifier import Wikifier

rfile = open("credentials.txt")
MY_KEY = rfile.readline()
rfile.close()

wik = Wikifier(MY_KEY)

TEXT = """
In the last section, we examined some early aspects of memory. In this section, what we’re going to do is discuss some factors that influence memory. So let’s do that by beginning with the concept on slide two, and that concept is overlearning. Basically in overlearning, the idea is that you continue to study something after you can recall it perfectly. So you study some particular topic whatever that topic is. When you can recall it perfectly, you continue to study it.
This is a classic way to help when one is taking comprehensive finals later in the semester. So when you study for exam one and after you really know it all, you continue to study it. That will make your comprehensive final easier.

The next factor that will influence memory relates to what we call organization. In general, if you can organize material, you can recall it better. There are lots of different types of organizational strategies and I’ve listed those on slide four. So let’s begin by talking about the first organizational strategy called clustering and is located on page five.

In clustering, basically you recall items better if you can recognize that there are two or more types of things in a particular list. So let’s give a couple of lists and show you some examples of that. These examples are shown in slide six.
"""	

def main(text):
    annotated_object = wik.wikify(text, 50, 50)
    # for i in annotated_object['annotation_data']:
    #     print(type(i['cosine']))
    print(annotated_object)


if __name__ == "__main__":
    main(TEXT)
