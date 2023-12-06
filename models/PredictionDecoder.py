import torch
import torch.nn as nn


class PredictionDecoder(torch.nn.Module):
    def __init__(self, dim_field, n_user: int = 60082, n_station: int = 14695, use_history=True, reflect_data=None,
                 type_all=None, dropout: float = 0.1):
        super().__init__()
        self.user_embedding = nn.Embedding(n_station, dim_field)
        self.station_embedding = nn.Embedding(n_user, dim_field)
        self.projection_layer = nn.Linear(dim_field, dim_field)

        self.dropout = nn.Dropout(dropout)
        self.theta = nn.Parameter(torch.rand(n_station, 1), requires_grad=True)
        self.alpha_fields = nn.Parameter(torch.rand(n_user, 1), requires_grad=True)
        self.n_fields = n_user
        self.softmax = torch.nn.Softmax(dim=0)
        self.fc_station = nn.Linear(dim_field, 1)
        self.fc_user = nn.Linear(dim_field, 1)
        self.type = type_all
        self.reflect = reflect_data
        self.mlp_his = nn.Sequential(
            nn.Linear(dim_field, dim_field // 2),
            nn.LeakyReLU(),
            nn.Linear(dim_field // 2, dim_field),
        )
        if self.use_hierarchy:
            self.gamma = nn.Parameter(torch.rand(n_user, 1), requires_grad=True)
            self.mlp = nn.ModuleDict(
                {t: nn.Sequential(
                    nn.Linear(dim_field, dim_field // 2),
                    nn.LeakyReLU(),
                    nn.Linear(dim_field // 2, dim_field),
                ) for t in self.type}
            )

            self.fc_field_1 = nn.Linear(
                (len(self.type) - 1), 1)

        else:

            self.mlp = nn.Sequential(
                nn.Linear(dim_field, dim_field // 2),
                nn.LeakyReLU(),
                nn.Linear(dim_field // 2, dim_field),
            )
        self.use_his = use_history

    def forward(self, user_embedding, station_embedding, nodes, user_id, raw_field_embed):
        """
            Param: x_com: torch.Tensor, shape (company_num, company_embed_dim),
                   x_fid: torch.Tensor, shape (field_num, field_embed_dim),
                   nodes: list, [(his_n, now_n),...]
            """
        batch_embedding = []
        # theta = self.softmax(self.theta)
        theta = self.theta
        user_mem_stat = (1 - theta[user_id]) * user_embedding + \
                        theta[user_id] * self.user_embedding(
            torch.tensor(user_id).to(user_embedding.device))  # combine company and fields
        # user_con = user_con.unsqueeze(1).expand(-1,self.n_fields,-1)
        user_mem_stat = self.dropout(user_mem_stat)
        for i, user_node in enumerate(nodes):
            # shape (user_item_num, item_embedding_dim)
            now_node = user_node[1]
            his_node = user_node[0]
            # all_node = his_node.extend(now_node)
            now_projected_fields = station_embedding[now_node]

            # 1: now_node: embed + proj; his_node: mlp + proj; other: proj
            # beta, tensor, (items_total, 1), indicator vector, appear item -> 1, not appear -> 0
            beta = user_embedding.new_zeros(self.n_fields, 1)
            beta[now_node] = 1
            # alpha_fields = self.softmax(self.alpha_fields)
            alpha_fields = self.alpha_fields
            if self.use_his:
                beta[his_node] = 1
                embed = (1 - beta * alpha_fields) * self.projection_layer(
                    self.station_embedding(torch.tensor(range(self.n_fields)).to(user_embedding.device)))

                # appear items: (1 - self.alpha) * origin + self.alpha * update, not appear items: origin
                embed[now_node, :] = embed[now_node, :] + alpha_fields[now_node] * now_projected_fields
                his_mlp_fields = self.mlp_his(raw_field_embed[his_node])
                # his_mlp_fields = self.mlp_his(raw_field_embed[his_node])
                embed[his_node, :] = embed[his_node, :] + alpha_fields[his_node] * his_mlp_fields

            else:
                embed = (1 - beta * alpha_fields) * self.projection_layer(
                    self.station_embedding(torch.tensor(range(self.n_fields)).to(user_embedding.device)))

                # appear items: (1 - self.alpha) * origin + self.alpha * update, not appear items: origin
                embed[now_node, :] = embed[now_node, :] + alpha_fields[now_node] * now_projected_fields

            station_output = self.fc_station(embed)
            user_output = self.fc_user(user_mem_stat[i])

            predict_com = station_output.squeeze() + user_output

            batch_embedding.append(predict_com)

        output = torch.stack(batch_embedding)
        # print(output.shape)
        return output
