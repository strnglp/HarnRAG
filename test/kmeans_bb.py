import fitz
import numpy as np
import argparse
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

from sklearn.cluster import MeanShift

def apply_meanshift_clustering(bounding_boxes, bandwidth=100):
    """
    Applies MeanShift clustering to the bounding boxes based on their horizontal positions (x0, x1).
    Removes bounding boxes that are too vertically distant within each cluster.
    
    Parameters:
        bounding_boxes: List of tuples (x0, y0, x1, y1, page_num)
        bandwidth: Controls the window size for MeanShift.
    
    Returns:
        List of tuples: [(x0, y0, x1, y1, page_num, cluster_label), ...]
    """
    # Use (x0, x1) for horizontal clustering
    coordinates = np.array([(x0, x1) for x0, _, x1, _, _ in bounding_boxes])
    meanshift = MeanShift(bandwidth=bandwidth)
    labels = meanshift.fit_predict(coordinates)
    
    clustered_boxes = [
        (x0, y0, x1, y1, page_num, label)
        for (x0, y0, x1, y1, page_num), label in zip(bounding_boxes, labels)
    ]
    
    # Adjust clusters to remove vertically distant outliers
    return filter_vertically_distant_boxes(clustered_boxes, max_vertical_distance=80)

def filter_vertically_distant_boxes(clustered_boxes, max_vertical_distance):
    """
    Filters vertically distant bounding boxes within each cluster.
    
    Parameters:
        clustered_boxes: List of tuples (x0, y0, x1, y1, page_num, cluster_label)
        max_vertical_distance: Maximum allowed vertical distance within a cluster.
    
    Returns:
        List of tuples: Filtered bounding boxes with adjusted clusters.
    """
    from collections import defaultdict

    clusters = defaultdict(list)
    for box in clustered_boxes:
        clusters[box[5]].append(box)  # Group by cluster_label (last element)

    filtered_boxes = []
    for label, boxes in clusters.items():
        # Sort boxes by their vertical positions (y0)
        boxes.sort(key=lambda b: b[1])  # Sort by y0

        # Remove vertically distant boxes
        filtered_cluster = [boxes[0]]  # Always keep the first box
        for i in range(1, len(boxes)):
            prev_y1 = filtered_cluster[-1][3]  # y1 of the last kept box
            cur_y0 = boxes[i][1]  # y0 of the current box
            if cur_y0 - prev_y1 <= max_vertical_distance:
                filtered_cluster.append(boxes[i])

        filtered_boxes.extend(filtered_cluster)
    
    return filtered_boxes


def dbscan_then_kmeans(bounding_boxes, eps=10, min_samples=1, num_columns=2):
    """
    Combines DBSCAN and KMeans for robust column clustering.
    - DBSCAN removes vertical outliers.
    - KMeans clusters the remaining bounding boxes into columns.
    """
    # Step 1: Use DBSCAN to filter out vertical outliers
    coordinates = np.array([(y0, y1) for _, y0, _, y1, _ in bounding_boxes])  # Only vertical positions
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='manhattan')  # Use vertical distance
    dbscan_labels = dbscan.fit_predict(coordinates)
    
    # Filter out bounding boxes labeled as outliers (-1)
    filtered_boxes = [
        box for box, label in zip(bounding_boxes, dbscan_labels) if label != -1
    ]
    
    if not filtered_boxes:
        raise ValueError("All bounding boxes were filtered as outliers.")
    
    # Step 2: Apply KMeans to cluster remaining bounding boxes into columns
    coordinates_kmeans = np.array([(x0, x1) for x0, _, x1, _, _ in filtered_boxes])  # Use horizontal positions
    kmeans = KMeans(n_clusters=num_columns, random_state=0)
    kmeans_labels = kmeans.fit_predict(coordinates_kmeans)
    
    # Return the final clustering results
    return [
        (*box, label)
        for box, label in zip(filtered_boxes, kmeans_labels)
    ]


def apply_kmeans_clustering(bounding_boxes, num_columns=2):
    coordinates = np.array([(x0, x1) for x0, _, x1, _, _ in bounding_boxes])
    kmeans = KMeans(n_clusters=num_columns, algorithm="elkan")
    labels = kmeans.fit_predict(coordinates)
    return [
        (x0, y0, x1, y1, page_num, label)
        for (x0, y0, x1, y1, page_num), label in zip(bounding_boxes, labels)
    ]


def apply_dbscan_clustering(bounding_boxes, eps=2, min_samples=2):
    # Use x-coordinates for clustering
    coordinates = np.array([(x0, x1) for x0, _, x1, _, _ in bounding_boxes])
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='manhattan')  # Manhattan for horizontal clustering
    labels = dbscan.fit_predict(coordinates)
    
    # Group bounding boxes by clusters, ignoring outliers (label == -1)
    clusters = {i: [] for i in set(labels) if i != -1}
    for box, label in zip(bounding_boxes, labels):
        if label != -1:  # Ignore outliers
            clusters[label].append(box)
    
    # Recalculate bounding boxes for each cluster
    final_result = []
    for label, boxes in clusters.items():
        for box in boxes:
            final_result.append((*box, label))
    
    return final_result

def apply_hierarchical_clustering(bounding_boxes, num_columns=3):
#    coordinates = np.array([(x0, x1) for x0, _, x1, _, _ in bounding_boxes])
    coordinates = np.array([(x0, y0, x1, y1) for x0, y0, x1, y1, _ in bounding_boxes])
    
    # Apply Agglomerative Clustering
    clustering = AgglomerativeClustering(n_clusters=num_columns, linkage='complete', metric='manhattan')
    labels = clustering.fit_predict(coordinates)
    
    # Assign clusters to bounding boxes
    return [
        (*box, label)
        for box, label in zip(bounding_boxes, labels)
    ]

def extract_bounding_boxes(pdf_path):
    doc = fitz.open(pdf_path)
    bounding_boxes = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")
        for block in blocks:
            x0, y0, x1, y1, *_ = block
            bounding_boxes.append((x0, y0, x1, y1, page_num))

    return bounding_boxes


def exclude_header_footer(bounding_boxes, page_height, header_threshold=0.09, footer_threshold=0.0):
    header_height, footer_height = page_height * header_threshold, page_height * footer_threshold
    return [
        (x0, y0, x1, y1, page_num)
        for x0, y0, x1, y1, page_num in bounding_boxes
        if y1 >= header_height and y0 <= page_height - footer_height
    ]



def box_intersection(box1, box2):
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2
    x0_int, y0_int = max(x0_1, x0_2), max(y0_1, y0_2)
    x1_int, y1_int = min(x1_1, x1_2), min(y1_1, y1_2)
    return (x0_int, y0_int, x1_int, y1_int) if x0_int < x1_int and y0_int < y1_int else None


def adjust_boxes_to_remove_overlap(box1, box2):
    intersection = box_intersection(box1, box2)

    if intersection is None:
        return box1, box2

    x0_int, y0_int, x1_int, y1_int = intersection

    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2

    if x0_1 < x0_int:
        box1 = (x0_1, y0_1, x0_int, y1_1)  # Shrink from the right

    if x0_2 < x0_int:
        box2 = (x0_int, y0_2, x1_2, y1_2)  # Shrink from the left

    if y0_1 < y0_int:
        box1 = (x0_1, y0_1, x1_1, y0_int)  # Shrink from the bottom

    if y0_2 < y0_int:
        box2 = (x0_2, y0_int, x1_2, y1_2)  # Shrink from the top

    return box1, box2


def adjust_for_column_separation(clustered_boxes):
    adjusted_boxes = []
    for i, box1 in enumerate(clustered_boxes):
        for j, box2 in enumerate(clustered_boxes[i + 1:], i + 1):
            adjusted_box1, adjusted_box2 = adjust_boxes_to_remove_overlap(box1[:4], box2[:4])
            adjusted_boxes.append((adjusted_box1[0], adjusted_box1[1], adjusted_box1[2], adjusted_box1[3], box1[4], box1[5]))
            adjusted_boxes.append((adjusted_box2[0], adjusted_box2[1], adjusted_box2[2], adjusted_box2[3], box2[4], box2[5]))
    return adjusted_boxes


def draw_column_bounding_boxes(pdf_path, adjusted_boxes, output_path):
    doc = fitz.open(pdf_path)
    columns = {}
    for x0, y0, x1, y1, page_num, label in adjusted_boxes:
        columns.setdefault(page_num, {}).setdefault(label, {'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1})
        columns[page_num][label].update({
            'x0': min(columns[page_num][label]['x0'], x0),
            'x1': max(columns[page_num][label]['x1'], x1),
            'y0': min(columns[page_num][label]['y0'], y0),
            'y1': max(columns[page_num][label]['y1'], y1),
        })
    for page_num, page_columns in columns.items():
        page = doc.load_page(page_num)
        for coords in page_columns.values():
            rect = fitz.Rect(coords['x0'], coords['y0'], coords['x1'], coords['y1'])
            page.draw_rect(rect, color=(0, 0, 1), width=2)
    doc.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Identify and cluster text blocks in a PDF")
    parser.add_argument("pdf_path", help="Path to the input PDF file")
    parser.add_argument("output_path", help="Path to save the output PDF with bounding boxes")
    args = parser.parse_args()

    bounding_boxes = extract_bounding_boxes(args.pdf_path)
    doc = fitz.open(args.pdf_path)
    page_height = doc[0].rect.height

    bounding_boxes = exclude_header_footer(bounding_boxes, page_height, 0.2, 0.2)
    clustered_boxes = apply_kmeans_clustering(bounding_boxes)
    adjusted_boxes = adjust_for_column_separation(clustered_boxes)
    draw_column_bounding_boxes(args.pdf_path, adjusted_boxes, args.output_path)


if __name__ == "__main__":
    main()
