from truelearn.preprocessing import Wikifier


# YOUR API KEY here.
# you could register at https://wikifier.org/register.html.
API_KEY = ""


# pylint: disable=missing-function-docstring
def main():
    sample_text = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor
incididunt ut labore et dolore magna aliqua. Ac odio tempor orci dapibus
ultrices in iaculis nunc sed. Adipiscing elit duis tristique sollicitudin nibh.
At lectus urna duis convallis. Tincidunt lobortis feugiat vivamus at. Morbi
quis commodo odio aenean sed adipiscing diam. Id venenatis a condimentum vitae
sapien pellentesque. Libero nunc consequat interdum varius sit amet mattis
vulputate enim. Urna nec tincidunt praesent semper feugiat nibh sed pulvinar. A
iaculis at erat pellentesque adipiscing commodo elit. Faucibus turpis in eu mi.
Enim ut tellus elementum sagittis vitae et leo duis. Aenean vel elit
scelerisque mauris pellentesque pulvinar. Consequat nisl vel pretium lectus
quam. Id eu nisl nunc mi ipsum faucibus vitae aliquet. Sed lectus vestibulum
mattis ullamcorper. Congue nisi vitae suscipit tellus mauris a. Dui ut ornare
lectus sit.
    Tortor at risus viverra adipiscing at in tellus integer feugiat. Varius vel
pharetra vel turpis nunc. Justo nec ultrices dui sapien. Elit pellentesque
habitant morbi tristique senectus et netus et malesuada. Tristique senectus et
netus et. Suspendisse in est ante in nibh mauris cursus. Donec adipiscing
tristique risus nec feugiat in fermentum posuere urna. In dictum non
consectetur a erat nam at lectus urna. Pellentesque pulvinar pellentesque
habitant morbi tristique. Condimentum id venenatis a condimentum.
    Egestas erat imperdiet sed euismod nisi porta lorem mollis aliquam. Justo
nec ultrices dui sapien eget. Urna molestie at elementum eu facilisis sed.
Tellus integer feugiat scelerisque varius morbi enim nunc faucibus. Tristique
sollicitudin nibh sit amet commodo. Sit amet facilisis magna etiam tempor orci
eu lobortis. Quam vulputate dignissim suspendisse in est ante. Eu consequat ac
felis donec et. Scelerisque purus semper eget duis at tellus at. Viverra nam
libero justo laoreet.
    Sagittis vitae et leo duis. Diam vel quam elementum pulvinar etiam non.
Ornare aenean euismod elementum nisi quis eleifend. Integer eget aliquet nibh
praesent. At urna condimentum mattis pellentesque id nibh. Nunc sed blandit
libero volutpat sed cras ornare. In vitae turpis massa sed elementum tempus. Mi
bibendum neque egestas congue quisque egestas diam. Enim tortor at auctor urna
nunc id cursus metus aliquam. Tincidunt eget nullam non nisi. At imperdiet dui
accumsan sit amet nulla facilisi morbi tempus. Nunc id cursus metus aliquam
eleifend mi in nulla. Eu feugiat pretium nibh ipsum consequat nisl vel. Tortor
dignissim convallis aenean et tortor at. Rhoncus aenean vel elit scelerisque
mauris pellentesque pulvinar.
    Tempus quam pellentesque nec nam aliquam sem et. Mattis ullamcorper velit
sed ullamcorper morbi tincidunt ornare. Commodo quis imperdiet massa tincidunt
nunc pulvinar. Volutpat lacus laoreet non curabitur gravida arcu. Condimentum
mattis pellentesque id nibh tortor id aliquet lectus proin. Adipiscing commodo
elit at imperdiet dui accumsan sit. Odio facilisis mauris sit amet massa vitae
tortor condimentum. Id consectetur purus ut faucibus pulvinar elementum
integer. Erat nam at lectus urna. Posuere urna nec tincidunt praesent. Quam
pellentesque nec nam aliquam sem et tortor consequat id. Ultrices mi tempus
imperdiet nulla malesuada pellentesque. Libero enim sed faucibus turpis in.
Diam maecenas ultricies mi eget mauris pharetra et ultrices neque. Nulla
malesuada pellentesque elit eget gravida cum. In nulla posuere sollicitudin
aliquam ultrices sagittis orci a scelerisque. Curabitur gravida arcu ac tortor
dignissim. Ridiculus mus mauris vitae ultricies leo integer.
    """

    wikifier = Wikifier(API_KEY)

    annotations = wikifier.wikify(sample_text)
    print(annotations)


if __name__ == "__main__":
    main()
