"""J7: streaming soak. One server with sidecar + team + streaming decodes at
least 100,000 tokens; zero channel errors (sidecar or stream). Reuses the x4
soak with the streaming environment and also fails on a stream error line.
"""
import x4_sidecar_soak as soak
from x7_stream_common import STREAM

if __name__ == "__main__":
    soak.ENV = STREAM
    soak.LABEL = "STREAM-SOAK"
    soak.TAG = "x7_soak"
    soak.BANNER = "streaming ENABLED"
    soak.main()
