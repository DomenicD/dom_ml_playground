from Queue import Queue

class NeuralNetworkUtils(object):
    @staticmethod
    def OutputBreadthTraversal(network, action):
        queue = Queue()
        processed = set()
        visited = {}
        for output in network.output_layer:
            queue.put(output)
            processed.add(output)

        while not queue.empty():
            node = queue.get()
            if not visited.get(node.id, False):
                action(node)
                visited[node.id] = True
                for inbound in node.in_connections:
                    sender = inbound.sender
                    if not sender in processed:
                        queue.put(sender)
                        processed.add(sender)