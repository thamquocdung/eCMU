import argparse
import yt_dlp
import os
from pathlib import Path

def download_audio_file(url, dst="data/audio"):
    ydl_opts = {'format': 'bestaudio/best',
                'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '192', }],
                'outtmpl': os.path.join(dst, "%(id)s.%(ext)s")}
    print("Downloading Audio ....")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        try:
            info_dict = info_dict['entries'][0]
        except Exception as e:
            info_dict['ext'] = 'wav'

        fn = ydl.prepare_filename(info_dict)
        if os.path.exists(fn):
            return fn, True
        else:
            parent_dir = os.path.dirname(fn)
            basename = os.path.basename(fn)[:-4]
            for file in os.listdir(parent_dir):
                if file.startswith(basename) and file.endswith('.wav'):
                    fn = os.path.join(parent_dir, file)
                    return fn, True
            return '', False

def main():
    parser = argparse.ArgumentParser("Download audio from url", add_help=True, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("url", type=str, help="URL")
    
    parser.add_argument(
        "--dst",
        type=str,
        default="data/audio",
        help="Output path to save downloaded audio files"
    )
    args = parser.parse_args()
    download_audio_file(url=args.url, dst=args.dst)


if __name__ == "__main__":
    main()
