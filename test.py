from pygmtsar import Stack

sbas = Stack('', drop_if_exists=True)
sbas.compute_reframe()