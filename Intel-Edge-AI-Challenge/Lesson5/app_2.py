import argparse
import cv2
from inference import Network

INPUT_STREAM = "pets.mp4"
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-m", help=m_desc, required=True)
    optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    optional.add_argument("-d", help=d_desc, default='CPU')
    args = parser.parse_args()

    return args


def infer_on_video(args):
    # Initialize the Inference Engine
    plugin = Network()

    # Load the network model into the IE
    plugin.load_model(args.m, args.d, CPU_EXTENSION)
    net_input_shape = plugin.get_input_shape()

    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)

    # Process frames until the video ends, or process is exited
    counter = 0
    incident_flag = False
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        
        counter += 1

        # Pre-process the frame
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        # Perform inference on the frame
        plugin.async_inference(p_frame)

        # Get the output of inference
        if plugin.wait() == 0:
            result = plugin.extract_output()
            ### TODO: Process the output
            incident_flag = assess_scene(result, counter, incident_flag)

        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    
def assess_scene(result, counter, incident_flag):
    '''
    Based on the determined situation, potentially send
    a message to the pets to break it up.
    '''
    if result[0][1] == 1 and not incident_flag:
        timestamp = counter / 30
        print("Log: Incident at {:.2f} seconds.".format(timestamp))
        print("Break it up!")
        incident_flag = True # not to constantly say "break it up!"
    elif result[0][1] != 1:
        incident_flag = False

    return incident_flag

def main():
    args = get_args()
    infer_on_video(args)


if __name__ == "__main__":
    main()
