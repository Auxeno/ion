Size of the arrays a checkpoint would hold, as a human-readable string.

Returns
-------
str
    Total bytes across `Param`, `Buffer` and plain array leaves, scaled to B, KB, MB
    or GB. This is the array payload `ion.checkpoint.save` writes, so the file on disk
    is a little larger for its header.

Examples
--------
>>> nn.MLP([64, 256, 10], key=key).disk_usage
'75 KB'
