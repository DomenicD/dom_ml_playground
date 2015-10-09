from Queue import Queue

class NetworkUtils(object):
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
                    receiver = inbound.receiver
                    if not receiver in processed:
                        queue.put(receiver)
                        processed.add(receiver)