def fuse_predictions(face_result,
                     voice_result,
                     spoof_result):

    if spoof_result["status"] != "Real":

        return {
            "emotion": "Invalid",
            "confidence": 0
        }

    return {
        "emotion": face_result["emotion"],
        "confidence": face_result["confidence"]
    }