# non maximum suppression
def nms(predicted_results, iou_thresh):
    nms_results = []
    for i in range(len(predicted_results)):
        overlapped = False
        for j in range(i+1, len(predicted_results)):
            if box_iou(predicted_results[i]["box"], predicted_results[j]["box"]) > iou_thresh:
                overlapped = True
                if predicted_results[i]["objectness"] > predicted_results[j]["objectness"]:
                    temp = predicted_results[i]
                    predicted_results[i] = predicted_results[j]
                    predicted_results[j] = temp
        if not overlapped:
            nms_results.append(predicted_results[i])
    return nms_results
