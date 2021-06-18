import sys
from process_images_script import ProcessImages

"""
NOTE: Before running for the first time, enter "panoptes configure" into the command line. Then, copy and paste
 the following for the username and password fields (NOTE: the password text will be hidden):
 username: macrodarkmatter
 password: khjPrkU286ro
"""

pi = ProcessImages(username="macrodarkmatter@gmail.com",
                   password="khjPrkU286ro",
                   project_id=11726,
                   workflow_id=14437)

pi.run()
