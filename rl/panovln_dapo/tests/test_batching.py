import unittest
import torch
from panovln_dapo.batching import collate_inputs


class PanoCollationTests(unittest.TestCase):
    def row(self,tokens,images):
        return dict(input_ids=torch.tensor([tokens]),attention_mask=torch.ones(1,len(tokens),dtype=torch.long),
                    mm_token_type_ids=torch.zeros(1,len(tokens),dtype=torch.long),
                    image_num_images=torch.tensor([images]),image_current_index=torch.tensor([images-1]),
                    image_erp_geometry=torch.tensor([[2.44346094,0.]]*images,dtype=torch.float32),
                    image_grid_thw=torch.ones(images,3,dtype=torch.long),
                    pixel_values=torch.ones(images*4,12),panovggt_pixel_values=torch.ones(1,1,3,2,2))

    def test_unequal_history_preserves_geometry_and_local_indices(self):
        a,b=self.row([1,2],1),self.row([3,4,5],3)
        result=collate_inputs([a,b],99)
        self.assertEqual(result['input_ids'].tolist(),[[99,1,2],[3,4,5]])
        self.assertEqual(result['attention_mask'].tolist(),[[0,1,1],[1,1,1]])
        self.assertEqual(result['image_num_images'].tolist(),[1,3])
        self.assertEqual(result['image_current_index'].tolist(),[0,2])
        self.assertTrue(torch.equal(result['image_erp_geometry'],torch.cat([a['image_erp_geometry'],b['image_erp_geometry']])))
        self.assertEqual(tuple(result['panovggt_pixel_values'].shape),(2,1,3,2,2))

    def test_rejects_unknown_fields_and_lowered_geometry(self):
        row=self.row([1],1);row['image_erp_geometry']=row['image_erp_geometry'].bfloat16()
        with self.assertRaisesRegex(ValueError,'FP32'):collate_inputs([row],0)
        row=self.row([1],1);row['position_ids']=torch.tensor([[0]])
        with self.assertRaisesRegex(ValueError,'Unsupported'):collate_inputs([row],0)
