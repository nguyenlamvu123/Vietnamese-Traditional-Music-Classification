import gradio as st
import argparse, json, os
from build import load_best_model
from utils import PROD_predict as main
from utils import AUDIO_FROM_USER


mod_name = tuple(range(3))
outmp4list = list()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    model_listobj: list = list()

    for mt in mod_name:
        model_obj = load_best_model(mt + 1)
        model_listobj.append(model_obj)

    parser.add_argument(
        '--listen',
        type=str,
        default='0.0.0.0',
        help='IP to listen on for connections to Gradio',
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default='8501',
        help='Port to run the server listener on',
    )
    args = parser.parse_args()
    server_port = args.server_port
    server_name = args.listen

    def foo(dir, mod):
        src_folder = os.path.dirname(dir[0])
        jso = main(dir, src_folder, AUDIO_FROM_USER, *model_listobj, gra=True)
        return json.dumps(jso, sort_keys=True, indent=4, ensure_ascii=False)

    with st.Blocks() as demo:
        # mod_ = st.Radio(
        #     mod_name,
        #     label="Select model to classification",
        #     value=mod_name[1],
        # )

        input = st.File(file_count="directory")
        files = st.Textbox()
        show = st.Button(value="classification")
        show.click(
            foo,
            [
                input,
                # mod_,
            ],
            files
        )

    demo.launch(server_port=server_port, server_name=server_name)
