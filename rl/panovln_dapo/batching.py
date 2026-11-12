"""Native padded Pano collation prototype; GPU parity required before activation."""
import torch

TEXT_KEYS={'input_ids','attention_mask','mm_token_type_ids'}
IMAGE_KEYS={'pixel_values','image_grid_thw','image_erp_geometry',
            'image_num_images','image_current_index','panovggt_pixel_values'}


def collate_inputs(rows, pad_token_id):
    if not rows: raise ValueError('Empty microbatch')
    keys=set(rows[0])
    if any(set(row)!=keys for row in rows): raise ValueError('Inconsistent input keys')
    if keys-(TEXT_KEYS|IMAGE_KEYS): raise ValueError(f'Unsupported batched inputs: {keys-(TEXT_KEYS|IMAGE_KEYS)}')
    if any(row['input_ids'].shape[0]!=1 for row in rows): raise ValueError('Expected one decision per input')
    width=max(row['input_ids'].shape[1] for row in rows)
    result={}
    for key in keys:
        if key in TEXT_KEYS:
            padded=[]
            for row in rows:
                tensor=row[key]
                fill=pad_token_id if key=='input_ids' else 0
                padded.append(torch.nn.functional.pad(tensor,(width-tensor.shape[1],0),value=fill))
            result[key]=torch.cat(padded,dim=0)
        else:
            if key=='image_erp_geometry' and any(row[key].dtype!=torch.float32 for row in rows):
                raise ValueError('ERP geometry must stay FP32')
            result[key]=torch.cat([row[key] for row in rows],dim=0)
    # These remain per-example LOCAL indices; never offset current image index
    # by the number of images in a preceding example.
    return result
